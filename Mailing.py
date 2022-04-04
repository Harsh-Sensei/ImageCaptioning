import ssl,smtplib

port =465
context=ssl.create_default_context()

def mail(message="Hi"):
    #print(message)
    sender_email="crossmodaldistill@gmail.com"
    receiver_email_list=["3502.stkabirdin@gmail.com"]#,"3499.stkabirdin@gmail.com"]

    with smtplib.SMTP_SSL("smtp.gmail.com",port,context=context) as server:
        server.login("crossmodaldistill@gmail.com","htgsensei55")
        for receiver_email in receiver_email_list:
            server.sendmail(sender_email,receiver_email,message)

if __name__ == "__main__":
    mail("Hi")