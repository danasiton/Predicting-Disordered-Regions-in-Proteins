import torch
import esm
from pre_process import preprocess_data
from tqdm import tqdm
import argparse

def main(input_file: str, output_file: str):
    # Load the ESM model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disables dropout for deterministic results
    sequence_representations = []

    # Preprocess the data
    print(f"Preprocessing data from {input_file}...")
    data_dic = preprocess_data(input_file)

    # Convert data to list of tuples (id, sequence)
    data = [(k, v[0]) for k, v in data_dic.items()]
    chunks = list(chunk_list(data, 1))
    for chunk in tqdm(chunks, desc="Processing chunks", unit="chunk"):
        batch_labels, batch_strs, batch_tokens = batch_converter(chunk)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        print("Extracting embeddings...")
        token_representations = []
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        print("Generating per-sequence representations...")

        for i, tokens_len in enumerate(tqdm(batch_lens, desc="Processing sequences", unit="seq")):
            sequence_representation = token_representations[i, 1:tokens_len - 1]
            sequence_representations.append({
                "id": chunk[i][0],
                "rep": sequence_representation.cpu(),
                "labels": data_dic[chunk[i][0]][1]
            })

        # Save the embeddings
        print(f"Saving embeddings to {output_file}...")
        torch.save(sequence_representations, output_file)
        print("Embeddings saved successfully!")


def chunk_list(list, chunk_size):
    for i in range(0, len(list), chunk_size):
        yield list[i:i + chunk_size]


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate protein embeddings using ESM2.")
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to the input JSON file with protein data."
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Path to save the output embeddings file."
    )

    args = parser.parse_args()
    main(input_file=args.input_file, output_file=args.output_file)