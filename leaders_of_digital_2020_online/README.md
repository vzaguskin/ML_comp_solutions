# Leaders of digital 2020 online part
This is a complete code for 6-th place solution for the online round of [2020 leaders of digital competition](https://cups.mail.ru/contests/leadersofdigital), ITS track(classification and segmentation of bacteria colony images)

The solution utilizes pytorch framework with the catalyst framework on top, efficientnet and resnet models, unet segmentation architecture and also some tricks to use the leak in data that was officially allowed for using.

The resulting score is 6.914 private LB out of possible 7(1. precision for each of 6 bacteria types in classifictaion task and 0.914 segmentation IoU)
