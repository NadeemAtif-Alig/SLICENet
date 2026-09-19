from model.slicenet import Network


def build_model(model_name, num_classes):
    if model_name == 'slicenet':
        return Network(num_classes=num_classes)

