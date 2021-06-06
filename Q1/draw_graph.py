import matplotlib.pyplot as plt


def plot_graph(epochs,loss,acc,file_name):
    train_loss,test_loss=loss
    train_acc,test_acc= acc
    x = [i for i in range(0, epochs, 1)]
    plt.plot(x,test_loss,label="Test")
    plt.plot(x,train_loss,label="Train")
    plt.title("Loss of Test set & Train set")
    plt.legend()
    file_ = "./graph_data/"+str(epochs)+"_Loss_"+file_name + ".png"
    plt.savefig(file_,dpi=300)
    plt.clf()

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(x, test_acc, label="Test")
    plt.plot(x, train_acc, label="Train")
    plt.title("Accuracy of Test set & Train set")
    plt.legend()
    file_ = "./graph_data/"+str(epochs)+"_Accuracy_"+file_name + ".png"
    plt.savefig(file_, dpi=300)
    plt.clf()