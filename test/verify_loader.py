import unittest
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.data.loaders import KITTIDataset


class TestKITTIDataset(unittest.TestCase):
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        self.root_dir = "data/kitti"
        self.img_size = 64
        self.batch_size = 4
        self.transform = transforms.Compose(
            [transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()]
        )

    def test_directory_exists(self):
        """
        Verify that the data directory actually exists.
        """
        self.assertTrue(
            os.path.exists(self.root_dir),
            f"Root directory '{self.root_dir}' does not exist. Please ensure data is downloaded.",
        )

    def test_dataset_initialization(self):
        """
        Test if the dataset initializes correctly and is not empty.
        """
        if not os.path.exists(self.root_dir):
            self.skipTest("Data directory missing, skipping initialization test.")

        dataset = KITTIDataset(
            root_dir=self.root_dir, sequence_length=1, transform=self.transform
        )

        # Check if dataset has items
        self.assertGreater(len(dataset), 0, "Dataset length is 0 (empty).")

    def test_dataloader_shape(self):
        """
        Test if the DataLoader yields batches with the expected shape.
        """
        if not os.path.exists(self.root_dir):
            self.skipTest("Data directory missing, skipping shape test.")

        dataset = KITTIDataset(
            root_dir=self.root_dir, sequence_length=1, transform=self.transform
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Get a single batch
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            self.fail("DataLoader failed to yield a batch.")

        # Define expected shape: (B, SeqLen, C, H, W)
        # Note: Adjust '3' if your dataset returns grayscale, or '1' if sequence length varies
        expected_shape = (self.batch_size, 1, 3, self.img_size, self.img_size)

        self.assertEqual(
            batch.shape,
            expected_shape,
            f"Shape mismatch! Expected {expected_shape}, got {batch.shape}",
        )

    def test_tensor_value_ranges(self):
        """
        Test if the tensor values are within the valid range [0, 1].
        """
        if not os.path.exists(self.root_dir):
            self.skipTest("Data directory missing, skipping value range test.")

        dataset = KITTIDataset(
            root_dir=self.root_dir, sequence_length=1, transform=self.transform
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        batch = next(iter(dataloader))

        # Check min/max values
        self.assertGreaterEqual(
            batch.min().item(), 0.0, "Found pixel values less than 0.0"
        )
        self.assertLessEqual(
            batch.max().item(), 1.0, "Found pixel values greater than 1.0"
        )


if __name__ == "__main__":
    # running with verbosity=2 gives more detailed output
    unittest.main(verbosity=2)
