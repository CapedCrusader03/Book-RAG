import React, { useState, useEffect, useRef } from "react";

function App() {
  const [messages, setMessages] = useState([
    {
      sender: "assistant",
      text: "Hello! Ingest your books using the sidebar or ask me a question about your documents.",
    },
  ]);
  const [input, setInput] = useState("");
  const [books, setBooks] = useState([]);
  const [selectedBook, setSelectedBook] = useState("");
  const [loading, setLoading] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  
  const chatEndRef = useRef(null);

  // Auto-scroll chat history
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Load books list on startup
  useEffect(() => {
    fetchBooks();
  }, []);

  const fetchBooks = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/api/books");
      if (response.ok) {
        const data = await response.json();
        setBooks(data.books || []);
      }
    } catch (err) {
      console.error("Failed to fetch books:", err);
    }
  };

  const handleSend = async (e) => {
    if (e) e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input;
    setInput("");
    
    // Add user message to thread
    setMessages((prev) => [...prev, { sender: "user", text: userMessage }]);
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userMessage,
          book_title: selectedBook || null,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setMessages((prev) => [
          ...prev,
          {
            sender: "assistant",
            text: data.answer,
            sources: data.sources || [],
          },
        ]);
      } else {
        const errorData = await response.json();
        setMessages((prev) => [
          ...prev,
          {
            sender: "assistant",
            text: `Error: ${errorData.detail || "Failed to retrieve answer from server."}`,
          },
        ]);
      }
    } catch (err) {
      console.error("Query request failed:", err);
      setMessages((prev) => [
        ...prev,
        {
          sender: "assistant",
          text: "Error: Unable to connect to the backend server. Make sure the FastAPI backend is running.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleIngest = async () => {
    if (ingesting) return;
    setIngesting(true);
    setStatusMessage("Scanning and parsing books in 'data/raw'...");

    try {
      const response = await fetch("http://127.0.0.1:8000/api/ingest", {
        method: "POST",
      });
      if (response.ok) {
        const data = await response.json();
        if (data.processed_books.length > 0) {
          setStatusMessage(`Ingested ${data.processed_books.length} book(s) into ${data.total_chunks} chunks.`);
        } else {
          setStatusMessage("No new books found. Place PDF/TXT files in 'data/raw'.");
        }
        fetchBooks(); // Refresh book list
      } else {
        setStatusMessage("Ingestion failed. Check backend logs.");
      }
    } catch (err) {
      console.error("Ingestion failed:", err);
      setStatusMessage("Error: Could not connect to API server.");
    } finally {
      setIngesting(false);
    }
  };

  // Inline citation component
  const Citations = ({ sources }) => {
    const [open, setOpen] = useState(false);
    if (!sources || sources.length === 0) return null;

    return (
      <div className="citations-container">
        <div className="citations-toggle" onClick={() => setOpen(!open)}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" style={{ transform: open ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s' }}>
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
          {open ? "Hide Citations" : `Show Citations (${sources.length})`}
        </div>
        
        {open && (
          <div className="citations-list">
            {sources.map((src, index) => {
              const meta = src.metadata || {};
              return (
                <div key={index} className="citation-card">
                  <div className="citation-meta">
                    <span>{meta.book_title || "Unknown Book"}</span>
                    {meta.page_number !== undefined && (
                      <span>Page {meta.page_number}</span>
                    )}
                    {meta.chunk_index !== undefined && (
                      <span>Chunk #{meta.chunk_index}</span>
                    )}
                  </div>
                  <div className="citation-text">
                    {src.text}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="app-container">
      {/* Sidebar Panel */}
      <aside className="sidebar">
        <div className="brand-section">
          <div className="brand-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
            </svg>
          </div>
          <span className="brand-name">Book RAG Explorer</span>
        </div>

        {/* Book filtering scope */}
        <div>
          <h3 className="section-title">Query Scope</h3>
          <select 
            className="select-input"
            value={selectedBook}
            onChange={(e) => setSelectedBook(e.target.value)}
          >
            <option value="">Search across all books</option>
            {books.map((book, idx) => (
              <option key={idx} value={book}>{book}</option>
            ))}
          </select>
        </div>

        {/* Database ingestion */}
        <div>
          <h3 className="section-title">Ingest Pipeline</h3>
          <button 
            className="btn-primary" 
            onClick={handleIngest}
            disabled={ingesting}
          >
            {ingesting ? (
              <>
                <div className="spinner"></div>
                Processing...
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                Scan & Ingest Books
              </>
            )}
          </button>
        </div>

        {/* Status display panel */}
        {(statusMessage || books.length > 0) && (
          <div className="status-card">
            <h3 className="section-title" style={{ margin: 0 }}>System Status</h3>
            {statusMessage && (
              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", wordBreak: "break-word" }}>
                {statusMessage}
              </p>
            )}
            <div className="status-row" style={{ marginTop: statusMessage ? "8px" : 0 }}>
              <span className="status-label">Ingested Books</span>
              <span className="status-value">{books.length}</span>
            </div>
            <div className="status-row">
              <span className="status-label">Database Status</span>
              <span className="status-value status-active">Online</span>
            </div>
          </div>
        )}
      </aside>

      {/* Chat Workspace */}
      <main className="workspace">
        <header className="chat-header">
          <h2 className="chat-header-title">
            {selectedBook ? `Querying: ${selectedBook}` : "Global Book Semantic Search"}
          </h2>
        </header>

        {/* Conversation flow */}
        <div className="chat-history">
          {messages.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.sender}`}>
              <div className="avatar">
                {msg.sender === "user" ? "U" : "AI"}
              </div>
              <div className="bubble">
                <p>{msg.text}</p>
                {msg.sender === "assistant" && msg.sources && (
                  <Citations sources={msg.sources} />
                )}
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="chat-message assistant">
              <div className="avatar">AI</div>
              <div className="bubble" style={{ display: "flex", alignItems: "center", gap: "8px", padding: "12px 18px" }}>
                <div className="spinner" style={{ borderTopColor: "var(--accent-primary)" }}></div>
                <span style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>Synthesizing answer...</span>
              </div>
            </div>
          )}
          
          <div ref={chatEndRef} />
        </div>

        {/* Input area */}
        <div className="chat-input-container">
          <form onSubmit={handleSend} className="chat-input-wrapper">
            <input
              type="text"
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about the books..."
              disabled={loading || ingesting}
            />
            <button 
              type="submit" 
              className="btn-send"
              disabled={!input.trim() || loading || ingesting}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;
