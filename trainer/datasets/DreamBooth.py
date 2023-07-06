import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            concepts_list,
            tokenizer,
            size=512,
            center_crop=False,
            num_class_images=None,
            pad_tokens=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.pad_tokens = pad_tokens

        self.instance_images_path = []
        self.class_images_path = []

        for concept in concepts_list:
            inst_img_path = [
                (x, concept["instance_prompt"])
                for x in Path(concept["instance_data_dir"]).iterdir()
                if x.is_file()
            ]
            self.instance_images_path.extend(inst_img_path)

            class_img_path = [
                (x, concept["class_prompt"])
                for x in Path(concept["class_data_dir"]).iterdir()
                if x.is_file()
            ]
            self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_path, instance_prompt = self.instance_images_path[
            index % self.num_instance_images
            ]
        instance_image = Image.open(instance_path).convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="max_length" if self.pad_tokens else "do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        class_path, class_prompt = self.class_images_path[index % self.num_class_images]
        class_image = Image.open(class_path).convert("RGB")
        example["class_images"] = self.image_transforms(class_image)
        example["class_prompt_ids"] = self.tokenizer(
            class_prompt,
            padding="max_length" if self.pad_tokens else "do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example
