def select_test_inh(data, inhibitor_name):
    train_data = data[~(data['Inhibitor'] == inhibitor_name)]
    test_data = data[data['Inhibitor'] == inhibitor_name]
    return train_data, test_data