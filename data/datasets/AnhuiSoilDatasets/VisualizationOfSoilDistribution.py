import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = 'AnHui.HuangShan.SOIL.csv'
df = pd.read_csv(file_path)

soil_content_columns = ['N', 'P', 'K']

fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=800)
fig.suptitle('Distributions of Soil Content Parameters in Anhui Soil Dataset', fontsize=18, y=1.08)

for i, param in enumerate(soil_content_columns):
    min_value, max_value = df[param].min(), df[param].max()
    sns.histplot(df[param], bins=20, kde=True, ax=axes[i], label='Histogram', color='grey', edgecolor='black')
    # sns.kdeplot(df[param], color='blue', linewidth=2, ax=axes[i], label='KDE')
    mean_value = df[param].mean()
    axes[i].axvline(mean_value, color='red', linestyle='--', linewidth=2, label='Mean')
    axes[i].text(mean_value + 0.01 * max_value, axes[i].get_ylim()[1] * 0.6, f'Mean: {mean_value:.2f}', color='red', fontsize=18)
    axes[i].set_xlim([min_value - (0.1 * max_value), max_value + (0.1 * max_value)])
    axes[i].set_title(f'Distribution of {param}', fontsize=20)
    axes[i].set_xlabel(f'{param} Content (mg.kg-1)', fontsize=18)
    axes[i].set_ylabel('Frequency', fontsize=22)
    axes[i].tick_params(labelsize=20)
    axes[i].grid(True, linestyle='--', linewidth=0.5)
    axes[i].legend(fontsize=20, loc='upper right')

plt.tight_layout()
plt.savefig('./Soil_Parameters_Distributions.png')
plt.show()
