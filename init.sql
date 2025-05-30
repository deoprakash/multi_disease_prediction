CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(100) NOT NULL UNIQUE,
  password VARCHAR(255) NOT NULL
);


CREATE TABLE uploads (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    filename VARCHAR(255),
    file_type VARCHAR(50),
    cloud_url TEXT,
    prediction TEXT,              -- ✅ Added prediction column
    uploaded_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id)
);