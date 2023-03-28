import util
import json
from argparse import ArgumentParser

if __name__ == "__main__":
    ap = ArgumentParser(description="Train")

    ap.add_argument(
        "image_path",
        help="Path to image",
    )
    ap.add_argument(
        "checkpoint",
        default="./checkpoint.pth",
        help="Path to checkpoint (default: checkpoint.pth)",
    )
    ap.add_argument(
        "--top_k",
        dest="top_k",
        action="store",
        default=5,
        type=int,
        help="Return top K most likely classes (default: 5)",
    )
    ap.add_argument(
        "--category_names",
        dest="category_names",
        action="store",
        default="./cat_to_name.json",
        help="Learning rate (default: ./cat_to_name.json)",
    )
    ap.add_argument(
        "--gpu",
        default=False,
        action="store_true",
        help="Enable GPU support for training",
    )

    args = ap.parse_args()

    # Load the datasets
    print("1. Load datasets...")
    data_loaders = util.load_data()

    # Load the pre-trained model
    print("2. Load pre-trained model...")
    model = util.load_checkpoint(args.checkpoint, args.gpu)

    # Run inference on image file
    print("3. Run inference on image file...")
    probs, classes = util.predict(args.image_path, model, args.top_k, args.gpu)

    print("Outputting results...")
    with open(args.category_names, "r") as json_file:
        cat_to_name = json.load(json_file, strict=False)

    # Print the top K classes with their probabilities
    for i in range(args.top_k):
        print(f"{cat_to_name[classes[i]]}: {probs[i] * 100:.3f}% probability")
