## Instructions

Model was trained on a GTX 1080Ti home server.

* Send a mail to ask database at this address: https://stanfordmlgroup.github.io/competitions/chexpert/
* Put the database in the folder `data` in the root directory
* Then do the following statements:

```bash
cd src

## Separate from database
python3 train_val_csv.py

## Training
python3 train.py --classes 3 --part cardio --network resent50
python3 train.py --classes 4 --part main --network resent101
python3 train.py --classes 4 --part nothing --network resent101
python3 train.py --classes 7 --part lungs --network resent101
python3 train.py --classes 4 --part pleural --network resent101

## Result
python3 inference.py
```