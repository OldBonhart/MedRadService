<a href="https://medradindustries.herokuapp.com/"><p align="center">
  <img border="0" alt="MedRadService" src="https://github.com/OldBonhart/MedRadService/blob/master/static/images/Xray_logo2.png" width="150" height="150"> </p>
<h1 align="center">
<strong>
  MedRadService<a>
  </strong></a>
</h1> <p align="center"></a>
 <strong> On the verge of defeating the tyranny of biology</strong>
</p>
<br>

<p align="center">
    <a href="https://www.youtube.com/watch?v=vmHD25Oyvj8"><img border="0" alt="MNedRadService" src="https://github.com/OldBonhart/MedRadService/blob/master/medradservice.gif" width="725" height="450"></a>
</p>

---

# Introductions

This is an example of the tandem of [**PyTorch**](https://pytorch.org/), [**Django**](https://docs.djangoproject.com/en/2.2/releases/2.0/) and [**Heroku.**](https://devcenter.heroku.com)<br>
This application was written as a practice.All you have to do is upload a chest x-ray, and then you will get a Grad-cam and prediction with probabilities.<br>
If you are interested in creating any interface for other people to interact with your ML-models, then this repository can be an example and starting point for this.

## About Bot's prediction models
**Blindness detection :**
+ This is Resnet18, trained on a dataset on [**NIH Chest X-ray Datasetfrom**](https://www.kaggle.com/nih-chest-xrays/data)
The predictive model of the service has a minimal configuration due to the limitations of the Heroku free server. <br>

---
# Notes on Heroku

#### [You have to check two important things when you want to send emails via Gmail smtp](https://stackoverflow.com/questions/36244309/heroku-not-sending-email-with-gmail-smtp):
1.Your apps configuration:
   + host: smtp.gmail.com
   + port: 587 or 465 (587 for tls, 465 for ssl)
   + protocol: tls or ssl
   + user: YOUR_USERNAME @ gmail.com
   + password: YOUR_PASSWORD
2. The given Gmail account settings:
  + If you've turned on 2-Step Verification for your account, you might need to enter an App password.
  + Without 2-Step Verification:
       1. Allow less secure apps access to your account.
       2. Visit http://www.google.com/accounts/DisplayUnlockCaptcha and sign in with your Gmail username and password.

