Shopping Occasions and Sequential Recommendation in E-commerce(WSDM2020)
============

## CuRe

Code of paper "[Time to Shop for Valentine’s Day: Shopping Occasions and Sequential Recommendation in E-commerce](http://people.tamu.edu/~jwang713/pubs/occasion-wsdm2020.pdf)".

![The proposed framework](framework.png)

## Datasets

The preprocessed datasets are included in the repo (`e.g. data/amazon/amazon_all.txt`), where each line contains an `user id` and `item id` (starting from 1) meaning an interaction (sorted by timestamp). Note that the last two interactions of each user are after the a global cutting time (please refer to Section 4.1 in our paper), acting as validation and testing case for the user.


## Requirements
python==3.6.8

tensorflow==1.14.0

## Usage
```python run.py --epochs 100```

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{wang2020time,
  title={Time to Shop for Valentine's Day: Shopping Occasions and Sequential Recommendation in E-commerce},
  author={Wang, Jianling and Louca, Raphael and Hu, Diane and Cellier, Caitlin and Caverlee, James and Hong, Liangjie},
  booktitle={Proceedings of the 13th International Conference on Web Search and Data Mining},
  pages={645--653},
  year={2020}
}
```
