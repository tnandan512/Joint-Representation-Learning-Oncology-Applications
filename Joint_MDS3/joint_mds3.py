import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joint_mds import JointMDS
import utils.scores as scores
import utils.mds as mds
from utils.utils import geodesic_dist, plot_embedding


def joint_mds3(
    core,
    B,
    C,
    n_components,
    alpha,
    eps,
    max_iter,
    dissimilarity,
    subplot=False,
    plot=False,
    labels=None,
    dataset=None,
    k_dict=None,
    k=None,
):
    """Performs Joint MDS on three modalities and evaluates alignment performance.

    Parameters:
        core, B, C (np.array): Input data matrices.
        n_components (int): Number of MDS components.
        alpha (float): Regularization parameter.
        eps (float): Convergence tolerance.
        max_iter (int): Maximum iterations for optimization.
        dissimilarity (str): Type of distance metric.
        subplot (bool): Whether to plot embeddings in subplots.
        plot (bool): Whether to generate a scatter plot.
        labels (np.array): Class labels for evaluation.
        dataset (str): Dataset name (for selecting k value if provided).
        k_dict (dict): Dictionary of k values per dataset.
        k (int): Neighborhood parameter for geodesic distances.

    Returns:
        Z1, Z2, Z3 (np.array): Aligned embeddings for core, B, and C.
    """

    np.random.seed(0)
    torch.manual_seed(0)

    # Select k value
    k = k_dict[dataset] if k_dict is not None else k

    def compute_alignment(source, target, name):
        """Helper function to perform Joint MDS alignment and evaluate scores."""
        D_source = torch.tensor(geodesic_dist(source, k=k), dtype=torch.float32)
        D_target = torch.tensor(geodesic_dist(target, k=k), dtype=torch.float32)

        jmds = JointMDS(
            n_components=n_components,
            alpha=alpha,
            eps=eps,
            max_iter=max_iter,
            dissimilarity=dissimilarity,
        )
        aligned_source, aligned_target, _, _ = jmds.fit_transform(D_source, D_target)

        aligned_source, aligned_target = aligned_source.numpy(), aligned_target.numpy()
        fracs = scores.calc_domainAveraged_FOSCTTM(aligned_source, aligned_target)

        print(f"Average FOSCTTM score for {name} alignment: {np.mean(fracs):.4f}")

        if labels is not None:
            acc = scores.transfer_accuracy(
                aligned_source, aligned_target, labels, labels, 5
            )
            print(f"Transfer accuracy for {name}: {acc:.4f}")

        return aligned_source, aligned_target

    # Perform alignments
    core1, BZ = compute_alignment(core, B, "core → B")
    core2, CZ = compute_alignment(core, C, "core → C")

    # Align core1 with core2
    jmds_final = JointMDS(
        n_components=n_components,
        alpha=alpha,
        eps=eps,
        max_iter=max_iter,
        dissimilarity="euclidean",
    )
    core1Z, core2Z, P, O = jmds_final.fit_transform(core1, core2)
    core1Z, core2Z = core1Z.numpy(), core2Z.numpy()

    fracs = scores.calc_domainAveraged_FOSCTTM(core1Z, core2Z)
    print(f"Average FOSCTTM score for core1 → core2 alignment: {np.mean(fracs):.4f}")

    if labels is not None:
        acc = scores.transfer_accuracy(core1Z, core2Z, labels, labels, 5)
        print(f"Transfer accuracy for core1 → core2: {acc:.4f}")

    # Align BZ and CZ to core subspace
    Z2 = np.dot(BZ, O.T)
    Z3 = np.dot(CZ, O.T)
    Z1 = core1Z

    # Visualization
    if subplot or plot:
        fig, axes = (
            plt.subplots(1, 3, figsize=(8, 4))
            if subplot
            else plt.subplots(figsize=(5, 4))
        )

        if subplot:
            domains = [("Core Domain", Z1), ("Domain 2", Z2), ("Domain 3", Z3)]
            for ax, (title, Z) in zip(axes, domains):
                plot_embedding(Z, labels - 1, ax, title)
                ax.set(xlabel="Joint MDS Component 1", ylabel="Joint MDS Component 2")
                ax.label_outer()
        else:
            plt.scatter(Z1[:, 0], Z1[:, 1], label="A")
            plt.scatter(Z2[:, 0], Z2[:, 1], label="B")
            plt.scatter(Z3[:, 0], Z3[:, 1], label="C")
            plt.legend()
            plt.xlabel("Joint MDS Component 1")
            plt.ylabel("Joint MDS Component 2")

        plt.show()

    return Z1, Z2, Z3
