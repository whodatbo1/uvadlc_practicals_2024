import argparse
import matplotlib.pyplot as plt
from train_mlp_pytorch import train

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    print('kwargs', kwargs)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    kwargs['use_batch_norm'] = True
    model_bn, val_accuracies_bn, test_accuracy_bn, logging_dict_bn = train(**kwargs)
    print('val_accuracies', val_accuracies)
    print('test_accuracy', test_accuracy)
    print('logging_dict', logging_dict)
    print('val_accuracies_bn', val_accuracies_bn)
    print('test_accuracy_bn', test_accuracy_bn)
    print('logging_dict_bn', logging_dict_bn)

    losses = [loss.detach().numpy() for loss in logging_dict['loss']]
    losses_bn = [loss.detach().numpy() for loss in logging_dict_bn['loss']]
    plt.figure()
    plt.plot(losses, label='loss')
    plt.plot(losses_bn, label='loss_bn')
    plt.grid(True)
    plt.legend()
    plt.title('Pytorch model: Loss for each epoch. Test accuracy {:.2f}. Batch test accuracy {:.2f}'.format(test_accuracy, test_accuracy_bn), wrap=True)
    plt.savefig('loss_pytorch.png')
    plt.show()

    val_accuracies = [acc.detach().numpy() for acc in logging_dict['val_accuracy']]
    val_accuracies_bn = [acc.detach().numpy() for acc in logging_dict_bn['val_accuracy']]

    plt.figure()
    plt.plot(val_accuracies, label='val_accuracy')
    plt.plot(val_accuracies_bn, label='val_accuracy_bn')
    plt.grid(True)
    plt.legend()
    plt.title('Pytorch model: Validation accuracy for each epoch. Test accuracy {:.2f}. Batch test accuracy {:.2f}'.format(test_accuracy, test_accuracy_bn), wrap=True)
    plt.savefig('val_accuracy_pytorch.png')
    plt.show()