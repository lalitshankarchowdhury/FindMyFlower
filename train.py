import util
from argparse import ArgumentParser

if __name__ == "__main__":
    ap = ArgumentParser(description="Train")

    ap.add_argument(
        "data_dir",
        help="Path to neural network training dataset",
    )
    ap.add_argument(
        "--save_dir",
        dest="save_dir",
        action="store",
        default="./checkpoint.pth",
        type=str,
        help="Checkpoint save path (default: ./checkpoint.pth)",
    )
    ap.add_argument(
        "--arch",
        dest="arch",
        action="store",
        default="vgg16",
        type=str,
        choices=["vgg11", "vgg13", "vgg16", "vgg19", "inception", "alexnet"],
        help="Pre-trained neural network architecture to use (default: vgg16)",
    )
    ap.add_argument(
        "--learning_rate",
        dest="learning_rate",
        action="store",
        default=0.001,
        type=float,
        help="Learning rate (default: 0.001)",
    )
    ap.add_argument(
        "--hidden_units",
        type=int,
        dest="hidden_units",
        action="store",
        default=4096,
        help="Number of hidden units (default: 4096)",
    )
    ap.add_argument(
        "--epochs",
        dest="epochs",
        action="store",
        type=int,
        default=5,
        help="Number of epochs to train the model for (default: 5)",
    )
    ap.add_argument(
        "--gpu",
        default=False,
        action="store_true",
        help="Enable GPU support for training",
    )

    args = ap.parse_args()

    print("1. Load datasets...")
    valid_dataset, data_loaders = util.load_data(args.data_dir)

    print("2. Setup model architecture...")
    model, criterion, optimizer = util.setup_model(
        architecture=args.arch,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        epochs=args.epochs,
        gpu=args.gpu,
    )
    print(model)

    print("3. Train model...")
    util.train_network(
        train_loader=data_loaders["train"],
        valid_loader=data_loaders["valid"],
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        gpu=args.gpu,
    )

    print("4. Validate test accuracy...")
    util.test_accuracy(data_loaders["test"], model, criterion, args.gpu)

    print("5. Save checkpoint to disk...")
    util.save_checkpoint(
        model=model,
        optimizer=optimizer,
        class_to_idx=valid_dataset.class_to_idx,
        architecture=args.arch,
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
