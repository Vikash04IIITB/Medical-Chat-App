import React from 'react';
import './ChatUI.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import '@fortawesome/fontawesome-free/css/all.min.css'; // Import Font Awesome

const ChatUI = ({ chatMessages, onUserMessage }) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    const userMessage = e.target.elements.message.value.trim();
    if (userMessage) {
      onUserMessage(userMessage);
      e.target.elements.message.value = '';
    }
  };

  return (
    <div className="card">
      <div className="card-header msg_head d-flex justify-content-between">
        <div className="d-flex bd-highlight">
          <div className="img_cont">
            <img
              src="https://www.prdistribution.com/spirit/uploads/pressreleases/2019/newsreleases/d83341deb75c4c4f6b113f27b1e42cd8-chatbot-florence-already-helps-thousands-of-patients-to-remember-their-medication.png"
              className="rounded-circle user_img"
              alt="User"
            />
            <span className="online_icon"></span>
          </div>
          <div className="user_info">
            <span>Medical Chatbot</span>
            <p>Ask me anything!</p>
          </div>
        </div>
      </div>
      <div id="messageFormeight" className="card-body msg_card_body">
        {chatMessages.map((message, index) => (
          <div
            key={index}
            className={`d-flex ${
              message.sender === 'user' ? 'justify-content-end mb-4' : 'justify-content-start mb-4'
            }`}
          >
            {message.sender === 'user' ? (
              <>
                <div className="msg_cotainer_send">{message.text}</div>
                <div className="img_cont_msg">
                  <img
                    src="https://i.ibb.co/d5b84Xw/Untitled-design.png"
                    className="rounded-circle user_img_msg"
                    alt="User"
                  />
                </div>
              </>
            ) : (
              <>
                <div className="img_cont_msg">
                  <img
                    src="https://www.prdistribution.com/spirit/uploads/pressreleases/2019/newsreleases/d83341deb75c4c4f6b113f27b1e42cd8-chatbot-florence-already-helps-thousands-of-patients-to-remember-their-medication.png"
                    className="rounded-circle user_img_msg"
                    alt="Bot"
                  />
                </div>
                <div className="msg_cotainer">{message.text}</div>
              </>
            )}
          </div>
        ))}
      </div>
      <div className="card-footer">
        <form id="messageArea" onSubmit={handleSubmit} className="input-group">
          <input
            type="text"
            id="text"
            name="message"
            placeholder="Type your message..."
            autoComplete="off"
            className="form-control type_msg"
            required
          />
          <div className="input-group-append">
            <button type="submit" id="send" className="input-group-text send_btn">
              <i className="fas fa-location-arrow"></i>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatUI;

