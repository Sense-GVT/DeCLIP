# Setup

Create two folders `imagenet_info` and `text_info` in the current project directory


```
/path/to/DeCLIP/
├── docs/
├── experiments/
├── linklink/
├── prototype/
├── text_info/
├── imagenet_info/
...
```



# Pretrain Dataset

## YFCC15M Setup

1. First Download our YFCC15M label file - [Google Driver](https://drive.google.com/file/d/1P-2_dHNc_c5XMY0A-89iNF5Cz_Y_Cfsy/view?usp=sharing) and put it into `imagenet_info` dir

2. Download Image data, You have two ways to download Image data:

+ DownLoad by labels: Crawl the image by the url in label dirctely.
+ Filter by label: Download offical [YFCC100M](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/) data, and Prepare the YFCC15M subset metadata pickle by the label.




## Text 


1. Download our vocab file for Text encoder [Google Driver](https://drive.google.com/file/d/1T9DMFiow_1KJpSmxbkQP0CubqCtLHNlG/view?usp=sharing)
2. put it into `text_info` dir


# Downstream Dataset

## Imagenet Setup

1. DownLoad offical ImageNet Dataset
2. DownLoad our ImageNet validation label file - [Google Driver](https://drive.google.com/file/d/1fgfjEzUwxEgLeOFon18kVvsOtp7KV7vh/view?usp=sharing) 
3. put it into `imagenet_info` dir
