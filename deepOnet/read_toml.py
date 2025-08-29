import tomllib  # Python 3.11+
# import tomli as tomllib  # for Python <=3.10

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

train_config = config["train"]
lr = train_config["lr"]
batchSize = train_config["batchSize"]
decayStepSize = train_config["decayStepSize"]
decayGamma = train_config["decayGamma"]

log_config = config['log']
log_on = log_config['on'] == 1
log_dir = log_config['dir']
log_train_legend = log_config['legend_train']
log_test_legend = log_config['legend_test']

print(f'LR: {lr} BS:{batchSize} SS:{decayStepSize} GM:{decayGamma} LOGON:{log_on}')
