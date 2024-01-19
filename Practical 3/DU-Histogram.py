import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(0)

# Example data
mu = 90  # Mean of distribution
sigma = 25  # Standard deviation of distribution
x = mu + sigma * np.random.randn(5000)

num_bins = 25
fig, ax = plt.subplots()

# Histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1)

# Add a 'best fit' line
y = stats.norm.pdf(bins, mu, sigma)
ax.plot(bins, y, '--')

ax.set_xlabel('Example Data')
ax.set_ylabel('Probability density')

# Title
sTitle = f'Histogram {len(x)} entries into {num_bins} Bins: $\mu={mu}$, $\sigma={sigma}$'
ax.set_title(sTitle)

fig.tight_layout()

sPathFig='C:/Users/prath/OneDrive/Desktop/Msc-IT Practicals/Data Science/Practical 3/DU-Histogram.png'
fig.savefig(sPathFig)
plt.show()
