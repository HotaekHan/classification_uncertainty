import yaml
import collections

def get_config(conf):
    with open(conf, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)

def print_config(conf):
    print(yaml.dump(conf, default_flow_style=False, default_style=''))

def _load_weights(weights_dict):
    key, value = list(weights_dict.items())[0]

    trained_data_parallel = False
    if key[:7] == 'module.':
        trained_data_parallel = True

    if trained_data_parallel is True:
        new_weights = collections.OrderedDict()
        for old_key in weights_dict:
            new_key = old_key[7:]
            new_weights[new_key] = weights_dict[old_key]
    else:
        new_weights = weights_dict

    return new_weights

if __name__ == '__main__':
    config_path = 'configs/test.yaml'

    cong = get_config(config_path)
    tmp=0