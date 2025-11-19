

def descriptive_stats(df):
    return df.describe()

def multi_histograms(df):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    cols = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'households']
    for i, col in enumerate(cols):
        axs[i].hist(df[col], bins=30, color='lightgreen')
        axs[i].set_title(col)
    fig.tight_layout()
    return fig

def price_distribution(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.histplot(df["median_house_value"], bins=40, kde=True, ax=ax, color='skyblue')
    ax.set_title("Distribution des prix des logements")
    return fig

def boxplot_price_by_location(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="ocean_proximity", y="median_house_value", ax=ax)
    ax.set_title("Prix médian par proximité de l'océan")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return fig

def correlation_heatmap(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Matrice de corrélation")
    return fig

def scatter_plot(df, x, y):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, alpha=0.3)
    ax.set_title(f"{y} en fonction de {x}")
    return fig

