import matplotlib.pyplot as plt


epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
training_accuracy = [.085671223, .0833417809, .0995, .108566412, .1278566412, .1965557512, .158566412, .16934901232, .14456341312, .10325]
training_cross_entropy = [14.3412131, 14.3131514, 10.3151231, 9.435141231, 9.14142423, 6.87421312, 9.314124512, 11.9842904, 11.123123, 13.546657]
training_f1_score = [0.0858791, 0.08642212, 0.091676663, 0.10989182, 0.12326731, 0.193459932, 0.16645643, 0.17111991, 0.14356131, 0.10713251]

validation_accuracy = [.076714323, .078458923, .099561231, .104512491, .127854162, .206522541, .14677777777, .1427777777, .126845134, .0918934312]
validation_cross_entropy = [14.2426123, 14.1239514, 12.98787135, 12.41287891, 11.3138086, 5.87421312, 9.41290972, 9.6546502, 11.4598051, 13.45156923]
validation_f1_score = [0.0758791, 0.08642312, 0.09122223, 0.09788882, 0.12323331, 0.1903451232, 0.137666643, 0.13111191, 0.123478131, 0.094765551]

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


