import os

from twilio.rest import Client

account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)


def send_message(message, to):
    message = client.messages.create(
        body=f"KAY/O: {message}",
        from_="+16592243401",
        to=to,
    )
    print(message.sid)


if __name__ == "__main__":
    send_message("Hello niko, I see you", "+17788141068")
