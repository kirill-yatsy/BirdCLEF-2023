import torch
import matplotlib.pyplot as plt


def train_loop(dataloader, model, criteria, optimizer, reshapeFn=None):
    size = len(dataloader.dataset)
    count = 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(reshapeFn(X) if reshapeFn else X)
        loss = criteria(pred, y)
        count += 1
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 128 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, reshapeFn=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for (X, y) in dataloader:
            pred = model(reshapeFn(X) if reshapeFn else X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def predict_and_show(model, loaderIterator):
    train_features, train_labels = next(loaderIterator)
    with torch.no_grad():
        img = torch.reshape(train_features[0], (28, 28))
        pred = model(train_features)
        label = torch.argmax(pred[0])
        plt.imshow(img)
        plt.title(f"predicted: {str(label)}, actual: {str(train_labels[0])}")
        plt.show()
