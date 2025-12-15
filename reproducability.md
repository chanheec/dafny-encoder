
# Reproducing Results

This document assumes that the reader already read both READMEs in each repository.




## Data Collection

Repo: https://github.com/chanheec/dafny-extractor

1. clone these Dafny project repositories.

https://github.com/dafny-lang/dafny  
https://github.com/vmware-labs/verified-betrfs  
https://github.com/mit-pdos/daisy-nfsd   
https://github.com/secure-foundations/veri-titan   
https://github.com/aws/aws-cryptographic-material-providers-library/   

2. for each of them, run the tool inside the above repo.


3. merge them into one large json using `concat-json.py`.

Please make sure to put `AWS Cryptographic Library` at last when you `concat-json.py`.
During training, we exclude them.
During evaluation, we use last 230 proofs (which is from the number of proofs in AWS Cryptographic Library)


## Training and evaluating model

Repo: https://github.com/chanheec/dafny-encoder

I trained and evaluated them on Google's Colab.
I uploaded the generated json file from the above step at my google drive,
and then I mounted it inside Colab.

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True )
```

Only part that needs to be updated is `Configuration & Setup` part that is at the top of each file.
In particular, paths (model, data, output) on each file needs to be updated before running them.

Run these sequentially.

1. (pre-training) `mlm.py`
2. (fine-tuning) `recommend.py`
3. (evaluation) `recommend-quant.py`




