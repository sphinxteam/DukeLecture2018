# Plot centroids
plt.plot(a_U[0, :] / np.sqrt(len(a_U)), a_U[1, :] / np.sqrt(len(a_U)), "kx", ms=10, mew=3)

# Plot data points with different colors
labels = np.array([np.argmax(a_V[i, :]) for i in range(len(a_V))])
for i in range(3):
    plt.plot(Y[0, labels == i], Y[1, labels == i], "o", ms=10, alpha=0.5)