# WeChatAutoJump

## Introduction
A program that can play the WeChat "Little Game" Jump Jump


## Method
1. first identify the little man and his direction to jump, by color match, noticing that the man has different color with all other game elements.

![src_withpos](./src_withpos_001.png) 

![color_match](./color_match_003.png) 

2. in the ROI, find the top of the next step by removing the background

![src_withroi](./src_withroi_002.png) 

![roi](./roi_005.png) 

![bgarea](./bgarea_004.png)


## Result

![screenshot](./Screenshot.png)
