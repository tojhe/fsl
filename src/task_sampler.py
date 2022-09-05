#!/usr/bin/python

import random
from typing import List, Tuple, Iterator

import torch as th
from torch.utils.data import Sampler, Dataset


class TaskSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
    ):
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks

        self.items_per_label = {}
        labels = dataset.targets
        for image_idx, label in enumerate(labels):
            label_image_idxes = self.items_per_label.get(label, [])
            label_image_idxes.append(image_idx)
            self.items_per_label[label] = label_image_idxes

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        
        for _ in range(self.n_tasks):

            task_sampled_images_list = []
            for label in random.sample(self.items_per_label.keys(), self.n_way):
                sampled_label_images = th.tensor(random.sample(self.items_per_label[label], self.n_shot + self.n_query))
                task_sampled_images_list.append(sampled_label_images)
            combined_task_samples = th.cat(task_sampled_images_list).tolist()

            yield combined_task_samples

            # [ [label_1_image-tensor, label_1_image-tensor, label_1_image-tensor],
            #  [label_2_image-tensor, label_2_image-tensor, label_2_image-tensor],
            #  [label_3_image-tensor, label_3_image-tensor, label_3_image-tensor], ]

    def episodic_collate_fn(
        self, 
        input_data: List[Tuple[th.Tensor, int]]
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, List[int]]:

        # Map unique classes
        unique_classes_index = list({sample[1] for sample in input_data})

        # Gather images
        all_images = th.cat([sample[0].unsqueeze(0) for sample in input_data])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )

        # Sample images for support and query
        support_images = all_images[:, : self.n_shot].reshape(
            (-1, *all_images.shape[2:])
        )
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))

        # Gather labels in task
        all_labels = th.tensor(
            [unique_classes_index.index(sample[1]) for sample in input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        # Allocate labels to respective samples in support and query
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            unique_classes_index,
        )