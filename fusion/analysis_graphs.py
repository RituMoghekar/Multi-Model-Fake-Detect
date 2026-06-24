import matplotlib.pyplot as plt

confidence_scores = [0.92, 0.85, 0.78, 0.55, 0.88, 0.91, 0.81]  # replace with real outputs

plt.figure()
plt.hist(confidence_scores, bins=5, edgecolor='black', linewidth=1.2)

plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Confidence Scores')

plt.savefig('graph5_confidence.png')
plt.show()