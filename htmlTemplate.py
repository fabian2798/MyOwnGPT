css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem display: flex
}
.chat-message.user {
    background-color: #87CEFA
}
.chat-message.bot {
    background-color: #79B30B
}
.chat-message .avatar {
    width 15%;
}
.chat-message .avatar img {
    max-width:78px;
    max-height:78px
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 85%;
    padding: 1.5rem;
    color: #fff
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="C:/Users/fabif/OneDrive/Desktop/Projects/chat_based_on_pdfs/svlfg-logo.svg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="C:/Users/fabif/OneDrive/Desktop/Projects/chat_based_on_pdfs/user-logo.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''