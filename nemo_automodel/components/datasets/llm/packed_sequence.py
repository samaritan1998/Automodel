# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
from datasets import Dataset, DatasetDict
from torch.nn import functional as F

logger = logging.getLogger(__name__)

CROSS_ENTROPY_IGNORE_IDX = -100
PACK_TYPE = dict[str, torch.Tensor | list[int]]


# based on https://github.com/pytorch/torchtune/blob/v0.6.1/torchtune/datasets/_packed.py#L17


def _fill_labels_with_cross_entropy_ignore_idx(labels: list[int], loss_mask: list[int]) -> list[int]:
    if len(loss_mask) != len(labels):
        raise ValueError(f"loss_mask and labels must have the same length, got {len(loss_mask)} vs {len(labels)}")
    for i, mask in enumerate(loss_mask):
        if mask == 0:
            labels[i] = CROSS_ENTROPY_IGNORE_IDX
    return labels


def _as_list(value) -> list[int]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value)


def _trim_sample_to_attention_mask(
    input_ids: list[int],
    labels: list[int],
    attention_mask,
) -> tuple[list[int], list[int], list[int], int]:
    """Drop right padding already present in a sample before packing."""
    if len(labels) != len(input_ids):
        raise ValueError(f"input_ids and labels must have the same length, got {len(input_ids)} vs {len(labels)}")
    if attention_mask is None:
        return input_ids, labels, [1] * len(input_ids), len(input_ids)

    attention_mask = [1 if int(x) != 0 else 0 for x in _as_list(attention_mask)]
    if len(attention_mask) < len(input_ids):
        attention_mask = attention_mask + [0] * (len(input_ids) - len(attention_mask))
    elif len(attention_mask) > len(input_ids):
        attention_mask = attention_mask[: len(input_ids)]

    actual_len = int(sum(attention_mask))
    expected_mask = [1] * actual_len + [0] * (len(input_ids) - actual_len)
    if attention_mask != expected_mask:
        raise ValueError("Packed samples require left-contiguous attention_mask; internal padding is unsupported.")

    if actual_len == 0:
        return [], [], [], 0
    return input_ids[:actual_len], labels[:actual_len], attention_mask[:actual_len], actual_len


def _pad_pack(
    pack: PACK_TYPE,
    padding_idx: int,
    packed_sequence_size: int,
    cross_entropy_ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    cp_size: int = 1,
    seq_pad_multiple: int = 1,
) -> PACK_TYPE:
    """
    Pads a pack to ``packed_sequence_size``.

    seq_lens contains original lengths.
    seq_lens_padded applies CP padding (if cp_size > 1) and pack-level padding.
    """
    # Pad tokens
    num_padding_tokens = packed_sequence_size - len(pack["input_ids"])
    padded_tokens = F.pad(
        pack["input_ids"],
        (0, num_padding_tokens),
        value=padding_idx,
    )

    # Pad labels
    padded_labels = F.pad(
        pack["labels"],
        (0, packed_sequence_size - len(pack["labels"])),
        value=cross_entropy_ignore_idx,
    )
    padded_attention_mask = F.pad(
        pack["attention_mask"],
        (0, packed_sequence_size - len(pack["attention_mask"])),
        value=0,
    )

    # seq_lens contains original sequence lengths
    original_seq_lens = pack["seq_lens"].clone()

    # seq_lens_padded applies per-sample alignment, then pack-level padding.
    # DSV4 packed CP uses seq_pad_multiple=128 so compressor windows do not
    # straddle packed-sample boundaries.
    align_multiple = max(2 * cp_size if cp_size > 1 else 1, int(seq_pad_multiple or 1))
    if align_multiple > 1:
        cp_padded_lens = []
        for seq_len in pack["seq_lens"]:
            cp_padded_len = ((seq_len + align_multiple - 1) // align_multiple) * align_multiple
            cp_padded_lens.append(cp_padded_len)

        # Convert to tensor
        padded_seq_lens = torch.tensor(cp_padded_lens, dtype=pack["seq_lens"].dtype, device=pack["seq_lens"].device)

        # Add pack-level padding to the last sequence
        if num_padding_tokens > 0 and len(padded_seq_lens) > 0:
            padded_seq_lens[-1] = padded_seq_lens[-1] + num_padding_tokens
    else:
        # No CP padding, just add pack-level padding to last sequence
        if num_padding_tokens > 0 and len(pack["seq_lens"]) > 0:
            padded_seq_lens = pack["seq_lens"].clone()
            padded_seq_lens[-1] = padded_seq_lens[-1] + num_padding_tokens
        else:
            padded_seq_lens = pack["seq_lens"].clone()

    # Pad position_ids continuing the sequence from last value
    # in position_ids
    # e.g. [0 1 2] -> [0 1 2 3 4 5] for packed_sequence_size = 6
    if len(pack["position_ids"]) == 0:
        raise ValueError("Cannot pad an empty packed sequence. Check packed_sequence_size and seq_pad_multiple.")
    num_range = torch.arange(
        pack["position_ids"][-1] + 1,
        pack["position_ids"][-1] + packed_sequence_size - len(pack["position_ids"]) + 1,
    )
    # Clamp to packed_sequence_size - 1 to avoid out of bounds error
    clamped_num_range = torch.clamp(num_range, 0, packed_sequence_size - 1)
    padded_position_ids = torch.cat([pack["position_ids"], clamped_num_range])

    padded_pack = {
        "input_ids": padded_tokens,
        "labels": padded_labels,
        "position_ids": padded_position_ids,
        "attention_mask": padded_attention_mask,
        "seq_lens": original_seq_lens,
        "seq_lens_padded": padded_seq_lens,
    }

    return padded_pack


def _convert_to_tensors(pack: PACK_TYPE) -> PACK_TYPE:
    """
    Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors.
    """
    tensor_pack = {
        "input_ids": torch.tensor(pack["input_ids"], dtype=torch.long),
        "labels": torch.tensor(pack["labels"], dtype=torch.long),
        "attention_mask": torch.tensor(pack["attention_mask"], dtype=torch.long),
        "position_ids": torch.tensor(pack["position_ids"], dtype=torch.long),
        "seq_lens": torch.tensor(pack["seq_lens"], dtype=torch.long),
    }
    return tensor_pack


def _tensorize_and_pad_pack(
    pack: PACK_TYPE,
    padding_idx: int,
    packed_sequence_size: int,
    cross_entropy_ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    cp_size: int = 1,
    seq_pad_multiple: int = 1,
) -> None:
    """
    converts to tensors, pads a pack and returns it.
    """
    pack = _convert_to_tensors(pack)
    pack = _pad_pack(
        pack,
        padding_idx=padding_idx,
        packed_sequence_size=packed_sequence_size,
        cross_entropy_ignore_idx=cross_entropy_ignore_idx,
        cp_size=cp_size,
        seq_pad_multiple=seq_pad_multiple,
    )
    return pack


def _should_stop_packing(max_packs: int, packs: list[PACK_TYPE]) -> bool:
    """
    If max packs is set, stop packing when we reach that number.
    """
    if max_packs is not None and len(packs) == max_packs:
        return True
    return False


def _split_and_add_pack(
    current_pack: PACK_TYPE,
    packs: list[PACK_TYPE],
    previous_sample_boundary: int,
    packed_sequence_size: int,
    padding_idx: int,
    cross_entropy_ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
    cp_size: int = 1,
    seq_pad_multiple: int = 1,
) -> PACK_TYPE:
    """
    Splits the current pack at the boundary, processes it, adds it to ``packs``.

    ...and returns the start of the next pack.

    TODO(@akoumparouli): refactor.
    """
    pack = {
        "input_ids": current_pack["input_ids"][:previous_sample_boundary],
        "labels": current_pack["labels"][:previous_sample_boundary],
        "attention_mask": current_pack["attention_mask"][:previous_sample_boundary],
        "position_ids": current_pack["position_ids"][:previous_sample_boundary],
        "seq_lens": current_pack["seq_lens"][:-1],
    }

    # Process and add the pack
    packs.append(
        _tensorize_and_pad_pack(
            pack,
            padding_idx=padding_idx,
            packed_sequence_size=packed_sequence_size,
            cross_entropy_ignore_idx=cross_entropy_ignore_idx,
            cp_size=cp_size,
            seq_pad_multiple=seq_pad_multiple,
        )
    )

    # Return the length of the last sample in the current pack
    next_seq_len = current_pack["seq_lens"][-1]

    output_dict = {
        "input_ids": current_pack["input_ids"][previous_sample_boundary:],
        "labels": current_pack["labels"][previous_sample_boundary:],
        "attention_mask": current_pack["attention_mask"][previous_sample_boundary:],
        "position_ids": current_pack["position_ids"][previous_sample_boundary:],
        "seq_lens": [next_seq_len],
    }
    return output_dict


def pack_dataset(
    dataset,
    split,
    packed_sequence_size,
    max_packs=None,
    padding_idx=0,
    drop_long_samples=True,
    cp_size=1,
    seq_pad_multiple: int = 1,
):
    """
    Pack the dataset to defined length.

    In particulat, it will iterate through the dataset. Use a buffer to hold samples until
    packed_sequence_size, then append the buffer to packs as a single "packed" sample.
    Continue until max_packs or end of dataset.

    Args:
        dataset: Actual dataset (can be 'train', 'val' or 'test')
        split (str): Whether the dataset is 'train', 'val' or 'test'
        packed_sequence_size (int): Number of tokens in a pack
        max_packs (int): Maximum number of packs. Default: None
        drop_long_samples (bool): If True, drop samples that are longer than packed_sequence_size.
        cp_size (int): Context parallel size. When > 1, each sequence will be padded to be
            divisible by 2*cp_size for context parallel processing. Default: 1 (no CP).
    """
    packs: list[PACK_TYPE] = []
    if isinstance(dataset, DatasetDict):
        if split in dataset:
            dataset = dataset[split]
        else:
            logger.warning(f"Dataset {split} not found. Using entire dataset.")

    # Buffer to hold samples until they are long enough to be added to packs
    current_pack = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "position_ids": [],
        "seq_lens": [],
    }
    previous_sample_boundary: int = 0
    logged_drop_long_samples = False

    # Calculate per-sample alignment. DSV4 packed CP sets seq_pad_multiple=128
    # so compressed/indexer windows cannot straddle two packed samples.
    cp_divisibility_factor = max(2 * cp_size if cp_size > 1 else 1, int(seq_pad_multiple or 1))
    if cp_divisibility_factor > 1 and packed_sequence_size % cp_divisibility_factor != 0:
        raise ValueError(
            f"packed_sequence_size ({packed_sequence_size}) must be divisible by the per-sample alignment "
            f"({cp_divisibility_factor}). Adjust packed_sequence_size or seq_pad_multiple."
        )

    for sample in dataset:
        input_ids, labels = _as_list(sample["input_ids"]), _as_list(sample["labels"])
        loss_mask = sample.get("loss_mask", None)
        if loss_mask is not None:
            labels = _fill_labels_with_cross_entropy_ignore_idx(labels, _as_list(loss_mask))
        input_ids, labels, attention_mask, seq_len = _trim_sample_to_attention_mask(
            input_ids,
            labels,
            sample.get("attention_mask"),
        )
        if seq_len == 0:
            continue
        # If the dataset outputs samples that are larger than the specified
        # packed_sequence_size and we're unable to split it, user needs to modify
        # one of the two parameters
        if drop_long_samples and seq_len > packed_sequence_size:
            if not logged_drop_long_samples:
                logged_drop_long_samples = True
                logger.info(f"Dataset has sampels longer than {packed_sequence_size}, will be skipped")
            continue

        if seq_len > packed_sequence_size:
            raise ValueError(
                f"Dataset sample is too long ({seq_len} > {packed_sequence_size}). "
                "Please increase `packed_sequence_size`.",
            )

        # Apply per-sample padding if needed.
        if cp_divisibility_factor > 1:
            cp_padded_len = ((seq_len + cp_divisibility_factor - 1) // cp_divisibility_factor) * cp_divisibility_factor
            cp_padding_amount = cp_padded_len - seq_len

            if cp_padding_amount > 0:
                # Add padding tokens
                input_ids = input_ids + [padding_idx] * cp_padding_amount
                labels = labels + [CROSS_ENTROPY_IGNORE_IDX] * cp_padding_amount
        padded_seq_len = len(input_ids)
        if drop_long_samples and padded_seq_len > packed_sequence_size:
            if not logged_drop_long_samples:
                logged_drop_long_samples = True
                logger.info(
                    "Dataset has samples whose aligned length exceeds %s; they will be skipped",
                    packed_sequence_size,
                )
            continue

        if padded_seq_len > packed_sequence_size:
            raise ValueError(
                f"Dataset sample is too long after alignment ({padded_seq_len} > {packed_sequence_size}; "
                f"original length {seq_len}, alignment {cp_divisibility_factor}). "
                "Increase `packed_sequence_size` or reduce `seq_pad_multiple`."
            )
        attention_mask = attention_mask + [0] * (len(input_ids) - len(attention_mask))

        # Update the current pack
        # "position_ids" is the pos ids, "seq_lens" is the len of each seq within the pack
        current_pack["input_ids"] += input_ids
        current_pack["labels"] += labels
        current_pack["attention_mask"] += attention_mask
        # Position IDs should continue for the actual length (including CP padding)
        current_pack["position_ids"] += [x % packed_sequence_size for x in range(len(input_ids))]
        # Always store original length in seq_lens
        current_pack["seq_lens"] += [seq_len]

        # If the current pack is over the packed_sequence_size, add it to packs and
        # retain any truncated or bumped samples for next pack
        while len(current_pack["input_ids"]) > packed_sequence_size and not _should_stop_packing(max_packs, packs):
            current_pack = _split_and_add_pack(
                current_pack,
                packs=packs,
                previous_sample_boundary=previous_sample_boundary,
                packed_sequence_size=packed_sequence_size,
                padding_idx=padding_idx,
                cross_entropy_ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
                cp_size=cp_size,
                seq_pad_multiple=seq_pad_multiple,
            )

        # Keep track of previous sample boundary
        previous_sample_boundary = len(current_pack["input_ids"])

        if _should_stop_packing(max_packs, packs):
            break

    # Handle the last pack if there's leftover and we haven't filled up the max packs
    if len(current_pack["input_ids"]) > 0 and (max_packs is None or len(packs) < max_packs):
        # No need to handle splitting at this point so we can just add the current pack
        packs.append(
            _tensorize_and_pad_pack(
                current_pack,
                padding_idx=padding_idx,
                packed_sequence_size=packed_sequence_size,
                cross_entropy_ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
                cp_size=cp_size,
                seq_pad_multiple=seq_pad_multiple,
            )
        )

    # After packing all samples, convert packs to a Dataset object
    logger.info("Total number of packs created: {}".format(len(packs)))
    if not packs:
        raise ValueError(
            f"No packs were produced: every sample was longer than packed_sequence_size={packed_sequence_size} "
            "and was dropped (drop_long_samples=True), or the input dataset was empty. "
            "Increase `packed_sequence_size` or provide samples that fit."
        )
    return Dataset.from_dict({key: [pack[key] for pack in packs] for key in packs[0].keys()})


def create_block_causal_mask(seq_lens: list[torch.Tensor]) -> torch.Tensor:
    """
    Creates causal mask block for specified lengths.

    In particular, given a batch tensor of seq lens defining the lengths of samples in each pack,
    Construct a 2D block causal mask for each pack in the batch. For example, if
    a single sample's seq_lens is [3, 2, 1], the mask would be::
        mask = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

    Args:
        seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.

    Returns:
        Tensor: Block causal mask of shape (batch_size, packed_sequence_size, packed_sequence_size).
    """
    batch_block_attn_masks = []
    batch_size = len(seq_lens)
    for sample_idx in range(batch_size):
        block_attn_masks = [
            torch.tril(
                torch.ones(
                    seq_len,
                    seq_len,
                    dtype=torch.bool,
                ),
            )
            for i, seq_len in enumerate(seq_lens[sample_idx])
        ]

        batch_block_attn_masks.append(torch.block_diag(*block_attn_masks))
    # Transformers expects the attn_mask to be 4d [bs, 1, packed_sequence_size, packed_sequence_size], hence adding
    # singleton (size 1) dimension at position 1.
    return torch.stack(batch_block_attn_masks).unsqueeze(1)


def packed_block_causal_mask(seq_lens: list[torch.Tensor]):
    """
    Create a 2D block causal document mask for a batch of packed sequences.

    Args:
        seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.

    Returns:
        _MaskType: BlockMask or Tensor if torch version < 2.5.0.
    """
    return create_block_causal_mask(seq_lens=seq_lens)
