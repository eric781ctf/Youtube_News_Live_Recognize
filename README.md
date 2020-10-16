# Youtube News Live Recognize
**News Channel - CTI、SET、TVBS、FTVN、EBC**

**Here's only the basic code without the training module of recognize the photo**

1.Capture the Picture from Youtube live stream(There are different scheme at each news channel).

2.Use trained modle to recognize whether the picture is title or not.

3.Upload the title picture to google api to identity the text in the picture.

---

The tool i used to recognize the graphics i captured from youtube are title or not : **pytorch**

The tool i used to recognize and turn the title in the graphic to string : **Google cloud vision**

I used thread to make sure it'll capture the graphic from those channels at the same time.

Author : Eric Chang
