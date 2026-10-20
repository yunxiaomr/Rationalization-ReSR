from typing import TypeVar, Set, Union, Dict, Callable

import torch as th

Tag = Union[str, Set[str]]

C = TypeVar("C", bound="BaseConfig")


class ConfigKey:
    def __init__(
        self,
        dataset: str,
        tags: Tag,
    ):
        self.dataset = dataset
        self.tags = {tags} if type(tags) == str else tags

    def __hash__(self) -> int:
        return hash((self.dataset, frozenset(self.tags)))

    def __str__(self) -> str:
        return f"{self.dataset}:{self.tags}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if other is None or not isinstance(other, ConfigKey):
            return False

        return self.dataset == other.dataset and self.tags == other.tags


class BaseConfig:
    configs: Dict[ConfigKey, str] = {}

    def __init__(
        self,
        seeds,
        num_classes,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=None,
        batch_size=8,
        freeze_embeddings: bool = False,
        use_pretrained_only: bool = False,
        add_sparsity_loss: bool = False,
        sparsity_level=0.15,
        sparsity_coefficient: float = 1.00,
        add_continuity_loss: bool = False,
        continuity_coefficient: float = 1.00,
        classification_coefficient: float = 1.00,
        max_epochs: int = 500,
        early_stopping_patience: int = 30,
    ):
        self.seeds = seeds
        self.num_classes = num_classes
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.batch_size = batch_size
        self.freeze_embeddings = freeze_embeddings
        self.use_pretrained_only = use_pretrained_only
        self.add_sparsity_loss = add_sparsity_loss
        self.sparsity_coefficient = sparsity_coefficient
        self.sparsity_level = sparsity_level
        self.add_continuity_loss = add_continuity_loss
        self.continuity_coefficient = continuity_coefficient
        self.classification_coefficient = classification_coefficient
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience

    @classmethod
    def from_config(cls, key: ConfigKey) -> C:
        config_method = cls.configs[key]
        return getattr(cls, config_method)()

    @classmethod
    def add_config(cls, key: ConfigKey, constructor: Callable[[], C]):
        setattr(cls, constructor.__name__, classmethod(constructor))
        cls.configs[key] = constructor.__name__
