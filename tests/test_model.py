import numpy as np
import torch

from catsdogs.evaluate import compute_confusion_matrix, most_confident_errors, per_class_metrics
from catsdogs.gradcam import GradCAM
from catsdogs.model import build_model, unfreeze_last_blocks


def test_build_model_output_shape():
    model = build_model(num_classes=2, pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_build_model_freezes_backbone_by_default():
    model = build_model(num_classes=2, pretrained=False)
    assert not any(p.requires_grad for p in model.layer4.parameters())
    assert all(p.requires_grad for p in model.fc.parameters())


def test_unfreeze_last_blocks():
    model = build_model(num_classes=2, pretrained=False)
    unfreeze_last_blocks(model, num_blocks=1)
    assert all(p.requires_grad for p in model.layer4.parameters())
    assert not any(p.requires_grad for p in model.layer3.parameters())


def test_compute_confusion_matrix_shape():
    labels = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 1, 1])
    cm = compute_confusion_matrix(labels, preds)
    assert cm.shape == (2, 2)
    assert cm.sum() == len(labels)


def test_per_class_metrics_perfect_predictions():
    labels = np.array([0, 1, 0, 1])
    metrics = per_class_metrics(labels, labels)
    assert np.allclose(metrics["precision"], [1.0, 1.0])
    assert np.allclose(metrics["recall"], [1.0, 1.0])


def test_most_confident_errors_ranks_by_confidence():
    labels = np.array([0, 0, 1])
    # example 0: correct; example 1: wrong, low confidence; example 2: wrong, high confidence
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.95, 0.05]])
    assert list(most_confident_errors(labels, probs, k=2)) == [2, 1]


def test_gradcam_output_shape():
    model = build_model(num_classes=2, pretrained=False)
    cam = GradCAM(model, model.layer4[-1])
    image = torch.randn(1, 3, 224, 224)
    heatmap = cam(image)
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
