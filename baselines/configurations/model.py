import torch as th

from configurations.base import BaseConfig, ConfigKey


DEFAULT_MAIN_SEEDS = [2023, 1337, 12345, 31415, 42]


class FRConfig(BaseConfig):
    configs = {
        ConfigKey(dataset="toy", tags={"spp", "fr"}): "toy",
        ConfigKey(dataset="hatexplain", tags={"spp", "fr"}): "hatexplain",
    }

    def __init__(
        self,
        embedding_dim,
        hidden_size,
        classification_head,
        selection_head,
        dropout_rate=0.1,
        temperature=1.0,
        embedding_name=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.classification_head = classification_head
        self.selection_head = selection_head
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.embedding_name = embedding_name

    @classmethod
    def toy(cls):
        return cls(
            seeds=DEFAULT_MAIN_SEEDS,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs={"lr": 1e-03},
            batch_size=64,
            embedding_dim=25,
            hidden_size=8,
            classification_head=lambda: th.nn.Sequential(th.nn.Linear(16, 3)),
            selection_head=lambda: th.nn.Sequential(th.nn.Linear(16, 2)),
            num_classes=3,
            dropout_rate=0.0,
            freeze_embeddings=True,
            add_sparsity_loss=True,
            sparsity_coefficient=1.0,
            sparsity_level=0.15,
            add_continuity_loss=True,
            continuity_coefficient=2.0,
            classification_coefficient=1.0,
        )

    @classmethod
    def hatexplain(cls):
        return cls(
            seeds=DEFAULT_MAIN_SEEDS,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs={"lr": 1e-03},
            batch_size=64,
            embedding_dim=25,
            hidden_size=16,
            classification_head=lambda: th.nn.Sequential(th.nn.Linear(32, 2)),
            selection_head=lambda: th.nn.Sequential(th.nn.Linear(32, 2)),
            num_classes=2,
            dropout_rate=0.0,
            freeze_embeddings=True,
            use_pretrained_only=True,
            embedding_name="twitter.27B",
            add_sparsity_loss=True,
            sparsity_coefficient=1.0,
            sparsity_level=0.22,
            add_continuity_loss=False,
            continuity_coefficient=1.0,
            classification_coefficient=1.0,
        )


class MGRConfig(BaseConfig):
    configs = {
        ConfigKey(dataset="toy", tags={"spp", "mgr"}): "toy",
        ConfigKey(dataset="hatexplain", tags={"spp", "mgr"}): "hatexplain",
    }

    def __init__(
        self,
        embedding_dim,
        hidden_size,
        classification_head,
        selection_head,
        num_generators,
        dropout_rate=0.1,
        temperature=1.0,
        embedding_name=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.classification_head = classification_head
        self.selection_head = selection_head
        self.num_generators = num_generators
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.embedding_name = embedding_name

    @classmethod
    def toy(cls):
        return cls(
            seeds=DEFAULT_MAIN_SEEDS,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs={"lr": 1e-03},
            batch_size=64,
            embedding_dim=25,
            hidden_size=8,
            classification_head=lambda: th.nn.Sequential(th.nn.Linear(16, 3)),
            selection_head=lambda: th.nn.Sequential(th.nn.Linear(16, 2)),
            num_classes=3,
            dropout_rate=0.0,
            freeze_embeddings=True,
            add_sparsity_loss=True,
            sparsity_coefficient=1,
            sparsity_level=0.15,
            add_continuity_loss=True,
            continuity_coefficient=2.0,
            classification_coefficient=1.0,
            num_generators=3,
        )

    @classmethod
    def hatexplain(cls):
        return cls(
            seeds=DEFAULT_MAIN_SEEDS,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs={"lr": 1e-03},
            batch_size=64,
            embedding_dim=25,
            hidden_size=16,
            classification_head=lambda: th.nn.Sequential(th.nn.Linear(32, 2)),
            selection_head=lambda: th.nn.Sequential(th.nn.Linear(32, 2)),
            num_classes=2,
            dropout_rate=0.0,
            freeze_embeddings=True,
            use_pretrained_only=True,
            embedding_name="twitter.27B",
            add_sparsity_loss=True,
            sparsity_coefficient=1,
            sparsity_level=0.22,
            add_continuity_loss=False,
            continuity_coefficient=1,
            classification_coefficient=1.0,
            num_generators=3,
        )


class MCDConfig(BaseConfig):
    configs = {
        ConfigKey(dataset="toy", tags={"spp", "mcd"}): "toy",
        ConfigKey(dataset="hatexplain", tags={"spp", "mcd"}): "hatexplain",
    }

    def __init__(
        self,
        embedding_dim,
        hidden_size,
        classification_head,
        selection_head,
        dropout_rate=0.1,
        temperature=1.0,
        embedding_name=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.classification_head = classification_head
        self.selection_head = selection_head
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.embedding_name = embedding_name

    @classmethod
    def toy(cls):
        return cls(
            seeds=DEFAULT_MAIN_SEEDS,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs={"lr": 1e-03},
            batch_size=64,
            embedding_dim=25,
            hidden_size=8,
            classification_head=lambda: th.nn.Sequential(th.nn.Linear(16, 3)),
            selection_head=lambda: th.nn.Sequential(th.nn.Linear(16, 2)),
            num_classes=3,
            dropout_rate=0.0,
            freeze_embeddings=True,
            add_sparsity_loss=True,
            sparsity_coefficient=1,
            sparsity_level=0.15,
            add_continuity_loss=True,
            continuity_coefficient=2.0,
            classification_coefficient=1.0,
        )

    @classmethod
    def hatexplain(cls):
        return cls(
            seeds=DEFAULT_MAIN_SEEDS,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs={"lr": 1e-03},
            batch_size=64,
            embedding_dim=25,
            hidden_size=16,
            classification_head=lambda: th.nn.Sequential(th.nn.Linear(32, 2)),
            selection_head=lambda: th.nn.Sequential(th.nn.Linear(32, 2)),
            num_classes=2,
            dropout_rate=0.0,
            freeze_embeddings=True,
            use_pretrained_only=True,
            embedding_name="twitter.27B",
            add_sparsity_loss=True,
            sparsity_coefficient=1,
            sparsity_level=0.22,
            add_continuity_loss=False,
            continuity_coefficient=1,
            classification_coefficient=1.0,
        )


class GRATConfig(BaseConfig):
    configs = {
        ConfigKey(dataset="toy", tags={"spp", "grat"}): "toy",
        ConfigKey(dataset="hatexplain", tags={"spp", "grat"}): "hatexplain",
    }

    def __init__(
        self,
        embedding_dim,
        hidden_size,
        classification_head,
        selection_head,
        dropout_rate=0.1,
        temperature=1.0,
        embedding_name=None,
        guide_coefficient: float = 1.00,
        jsd_coefficient: float = 1.00,
        pretrain_epochs: int = 10,
        guide_decay: float = 1e-04,
        noise_sigma: float = 1.00,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.classification_head = classification_head
        self.selection_head = selection_head
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.embedding_name = embedding_name
        self.guide_coefficient = guide_coefficient
        self.jsd_coefficient = jsd_coefficient
        self.pretrain_epochs = pretrain_epochs
        self.guide_decay = guide_decay
        self.noise_sigma = noise_sigma

    @classmethod
    def toy(cls):
        return cls(
            seeds=DEFAULT_MAIN_SEEDS,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs={"lr": 1e-03},
            batch_size=64,
            embedding_dim=25,
            hidden_size=8,
            classification_head=lambda: th.nn.Sequential(th.nn.Linear(16, 3)),
            selection_head=lambda: th.nn.Sequential(th.nn.Linear(16, 2)),
            num_classes=3,
            dropout_rate=0.0,
            freeze_embeddings=True,
            add_sparsity_loss=True,
            sparsity_coefficient=1,
            sparsity_level=0.15,
            add_continuity_loss=True,
            continuity_coefficient=2.0,
            classification_coefficient=1.0,
            guide_coefficient=1.0,
            jsd_coefficient=1.0,
            guide_decay=1e-05,
            pretrain_epochs=10,
            noise_sigma=1.00,
        )

    @classmethod
    def hatexplain(cls):
        return cls(
            seeds=DEFAULT_MAIN_SEEDS,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs={"lr": 1e-03},
            batch_size=64,
            embedding_dim=25,
            hidden_size=16,
            classification_head=lambda: th.nn.Sequential(th.nn.Linear(32, 2)),
            selection_head=lambda: th.nn.Sequential(th.nn.Linear(32, 2)),
            num_classes=2,
            dropout_rate=0.0,
            freeze_embeddings=True,
            use_pretrained_only=True,
            embedding_name="twitter.27B",
            add_sparsity_loss=True,
            sparsity_coefficient=1,
            sparsity_level=0.22,
            add_continuity_loss=False,
            continuity_coefficient=1,
            classification_coefficient=1.0,
            guide_coefficient=2.5,
            jsd_coefficient=1.5,
            guide_decay=1e-05,
            pretrain_epochs=10,
            noise_sigma=1.00,
        )
