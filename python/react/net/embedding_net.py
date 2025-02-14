import torch
from torch import nn
from timm import create_model


class EmbeddingNet(nn.Module):
    """
    Embedding network to generate visual embeddings for objects

    Attributes:
        backbone: The backbone network, in our work we used EfficientNet-B2
    """

    def __init__(self, model: nn.Module, remove_layers: int = 1):
        super().__init__()
        self.backbone = self._build_model(model=model, remove_layers=remove_layers)

    def _build_model(self, model: nn.Module, remove_layers: int):
        backbone = nn.Sequential(
            *list(model.children())[:-remove_layers],
        )
        return backbone

    def forward(self, img: torch.Tensor):
        return self.backbone(img)

    def num_features(self) -> int:
        return list(self.backbone.children())[-2].num_features


def get_embedding_model(
    weights: str,
    backbone: str,
) -> EmbeddingNet:
    embedding_model = EmbeddingNet(model=create_model(backbone))
    embedding_model.load_state_dict(state_dict=torch.load(weights, weights_only=True))
    embedding_model.cuda()
    embedding_model.eval()

    # Warm up
    embedding_model(torch.rand([1, 3, 224, 224]).cuda())
    return embedding_model
