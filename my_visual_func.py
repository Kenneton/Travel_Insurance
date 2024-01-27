import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import learning_curve

palette = "muted"
color = "blue"
custom_colors = sns.color_palette(palette, 2)  # for barcharts


# FORMATTING FUNCTIONS
def thousands_formatter(x, pos=None, decimals=0):
    return f"{x / 1000:.{decimals}f}k"


def millions_formatter(x, pos=None, decimals=0):
    return f"{x * 1e-6:.{decimals}f}M"


def camel_case_to_title(s):
    """
    Converts a camel case string to title format with the first word also capitalized.

    Parameters:
    - s: A string in camel case format.

    Returns:
    A string converted to title format.

    Example usage:
    print(camel_case_to_title("GraduateOrNot"))
    """
    s_with_spaces = re.sub(r"(?<!^)([A-Z])(?=[a-z])", r" \1", s)
    title_format = s_with_spaces.title()

    return title_format


# VISUALIZATION FUNCTIONS
def plot_barchart(
    df: pd.DataFrame,
    order: bool = False,
    percent: bool = True,
    scale: str = "as_is",
    decimals: int = 0,
) -> None:
    """
    Plots a barchart with low ink-to-data ratio from a dataframe.
    Annotates the values and percentages (optional).
    Formats large values to millions or thousands based on 'scale'.
    """

    if order:
        df = df.sort_values(by=df.columns[1], ascending=False)

    g = sns.barplot(
        data=df, x=df.columns[0], y=df.columns[1], palette="muted", width=0.7
    )

    # Update formatter_func to handle decimals when scale is 'as_is'
    formatter_func = lambda x: f"{x:.{decimals}f}"
    if scale == "millions":
        formatter_func = lambda x: millions_formatter(x, decimals=decimals)
    elif scale == "thousands":
        formatter_func = lambda x: thousands_formatter(x, decimals=decimals)

    for index, value in enumerate(df[df.columns[1]]):
        annotation = formatter_func(value)
        if percent:
            total = df[df.columns[1]].sum()
            percentage = f"({100 * value / total:.1f}%)"
            annotation = f"{annotation}\n{percentage}"

        plt.text(
            index,
            value,
            annotation,
            horizontalalignment="center",
            verticalalignment="bottom",
            color="black",
        )

    sns.despine(left=True, bottom=True)
    plt.yticks([])
    g.set_xlabel("")
    g.set_ylabel("")


def plot_stacked_bar_chart(
    df, category_col, target_col="TravelInsurance", figsize=(8, 6)
):
    """
    Plots a stacked bar chart with counts, percentages for the specified categorical column against a target column,
    and annotations for the combined stacked bars.

    Parameters:
    - df: pandas DataFrame containing the data.
    - category_col: String, the name of the categorical column to plot.
    - target_col: String, the name of the target column to plot against.
    - figsize: Tuple, the figure size.
    """

    ct = pd.crosstab(df[category_col], df[target_col])
    ax = ct.plot(
        kind="bar", stacked=True, figsize=figsize, color=custom_colors, alpha=0.7
    )
    ax.spines[["right", "top"]].set_visible(False)
    plt.title(f"{category_col}")
    plt.xlabel(category_col)
    plt.ylabel("")
    plt.xticks(rotation=0)
    plt.legend(title=target_col, labels=["No", "Yes"])

    bar_totals = ct.sum(axis=1)
    grand_total = bar_totals.sum()

    # Annotate the individual segments of the stacked bars
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        total = bar_totals[int(x + width / 2)]
        percentage = f"({height / total:.1%})" if total else ""

        ax.text(
            x + width / 2,
            y + height / 2,
            f"{int(height)} {percentage}",
            ha="center",
            va="center",
            fontsize=10,
        )

    # Annotate total bar values
    for i, total in enumerate(bar_totals):
        percentage_of_grand_total = f"({total / grand_total:.1%})"
        ax.text(
            i,
            total,
            f"{int(total)} {percentage_of_grand_total}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples)
        Target relative to X for classification or regression.
    axes : axes object, default (None)
        Axes used to plot the curve.
    ylim : tuple of shape (2,), default (None)
        Defines minimum and maximum y-values plotted.
    cv : int, cross-validation generator or an iterable, default (None)
        Determines the cross-validation splitting strategy.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve.
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(16, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color=custom_colors[0],
    )
    axes.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color=custom_colors[1],
    )
    axes.plot(
        train_sizes,
        train_scores_mean,
        "o-",
        color=custom_colors[0],
        label="Training score",
    )
    axes.plot(
        train_sizes,
        test_scores_mean,
        "o-",
        color=custom_colors[1],
        label="Cross-validation score",
    )
    axes.legend(loc="best")

    return plt
