import matplotlib.pyplot as plt


epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
training_accuracy = [.055671223, .0633417809, .0695, .088566412, .0978566412, .1065557512, .08578566412, .08934901232, .09456341312, .10325]
training_cross_entropy = [13.3412131, 12.3131514, 11.3151231, 12.435141231, 13.14142423, 14.87421312, 15.314124512,14.9842904, 13.123123, 12.546657]
training_f1_score = [0.0558791, 0.06642212, 0.071676663, 0.08989182, 0.09326731, 0.103459932, 0.117645643, 0.11111991, 0.11356131, 0.10713251]

validation_accuracy = [.096714323, .098458923, .089561231, .104512491, .107854162, .136522541, .12677777777, .1267777777, .136845134, .158934312]
validation_cross_entropy = [15.2426123, 13.1239514, 17.98787135, 15.41287891, 13.3138086, 13.87421312, 10.41290972, 13.6546502, 12.4598051, 9.45156923]
validation_f1_score = [0.0458791, 0.07642312, 0.07122223, 0.08788882, 0.09323331, 0.103451232, 0.107666643, 0.10111191, 0.103478131, 0.14765551]

plt.plot(epoch, training_accuracy)
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy vs Epoch')
plt.savefig('training_accuracy_vs_epoch.png')
# clear the plot
plt.clf()


# new plot
plt.plot(epoch, training_cross_entropy)
plt.xlabel('Epoch')
plt.ylabel('Training Cross Entropy')
plt.title('Training Cross Entropy vs Epoch')
plt.savefig('training_cross_entropy_vs_epoch.png')
plt.clf()
# new plot
plt.plot(epoch, training_f1_score)
plt.xlabel('Epoch')
plt.ylabel('Training F1 Score')
plt.title('Training F1 Score vs Epoch')
plt.savefig('training_f1_score_vs_epoch.png')
plt.clf()
# new plot
plt.plot(epoch, validation_accuracy)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs Epoch')
plt.savefig('validation_accuracy_vs_epoch.png')
plt.clf()

# new plot
plt.plot(epoch, validation_cross_entropy)
plt.xlabel('Epoch')
plt.ylabel('Validation Cross Entropy')
plt.title('Validation Cross Entropy vs Epoch')
plt.savefig('validation_cross_entropy_vs_epoch.png')
plt.clf()

# new plot
plt.plot(epoch, validation_f1_score)
plt.xlabel('Epoch')
plt.ylabel('Validation F1 Score')
plt.title('Validation F1 Score vs Epoch')
plt.savefig('validation_f1_score_vs_epoch.png')
plt.clf()


