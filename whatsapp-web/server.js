// WhatsApp Web.js wrapper service for Herinneringen.
// Provides REST endpoints for QR auth, phone pairing, sending messages, and listing contacts.
// Runs on port 3001 inside Docker, reachable from n8n on ai-net.

const express = require('express');
const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3001;
const DATA_PATH = '/data/session';
const PUPPETEER_ARGS = [
  '--no-sandbox',
  '--disable-setuid-sandbox',
  '--disable-dev-shm-usage',
  '--disable-gpu',
  '--single-process'
];

let client = null;
let qrData = null;
let ready = false;
let lastError = null;
let pairingCode = null;

function createClient(phoneNumber) {
  const opts = {
    authStrategy: new LocalAuth({ dataPath: DATA_PATH }),
    puppeteer: { headless: true, args: PUPPETEER_ARGS }
  };
  if (phoneNumber) {
    opts.pairWithPhoneNumber = {
      phoneNumber,
      showNotification: true
    };
  }
  return new Client(opts);
}

function wireEvents(c) {
  c.on('qr', (qr) => {
    qrData = qr;
    console.log('QR code received, waiting for scan...');
  });

  c.on('code', (code) => {
    pairingCode = code;
    console.log('Pairing code received:', code);
  });

  c.on('ready', () => {
    ready = true;
    qrData = null;
    pairingCode = null;
    lastError = null;
    console.log('WhatsApp client ready');
  });

  c.on('disconnected', (reason) => {
    ready = false;
    lastError = reason;
    console.warn('WhatsApp disconnected:', reason);
  });

  c.on('auth_failure', (msg) => {
    lastError = msg;
    console.error('WhatsApp auth failure:', msg);
  });
}

// Health check
app.get('/health', (_req, res) => {
  res.json({ status: 'ok', ready });
});

// Connection status
app.get('/status', (_req, res) => {
  res.json({ ready, hasQr: !!qrData, hasPairingCode: !!pairingCode, error: lastError });
});

// QR code for authentication
app.get('/qr', async (_req, res) => {
  if (ready) return res.json({ ready: true, qr: null });
  if (!qrData) return res.json({ ready: false, qr: null, message: 'Wachten op QR code...' });

  try {
    const dataUrl = await qrcode.toDataURL(qrData, { width: 256 });
    res.json({ ready: false, qr: dataUrl });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Request pairing code (reinitializes client with phone number)
app.post('/pair', async (req, res) => {
  if (ready) return res.json({ ready: true, code: null });

  const { phoneNumber } = req.body;
  if (!phoneNumber) {
    return res.status(400).json({ error: 'phoneNumber is verplicht (bijv. "31612345678")' });
  }

  try {
    // Destroy existing client and reinitialize with phone pairing
    if (client) {
      await client.destroy().catch(() => {});
    }

    qrData = null;
    pairingCode = null;
    lastError = null;
    ready = false;

    client = createClient(phoneNumber);
    wireEvents(client);

    // Wait for the 'code' event (up to 30s)
    const code = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Timeout: geen pairing code ontvangen')), 30000);
      client.once('code', (c) => {
        clearTimeout(timeout);
        resolve(c);
      });
      client.initialize().catch((err) => {
        clearTimeout(timeout);
        reject(err);
      });
    });

    const formatted = code.length === 8
      ? `${code.slice(0, 4)}-${code.slice(4)}`
      : code;
    res.json({ code: formatted });
  } catch (e) {
    console.error('Pairing code request failed:', e);
    lastError = e.message;
    res.status(500).json({ error: e.message });
  }
});

// Send a message
app.post('/send', async (req, res) => {
  if (!ready) return res.status(503).json({ error: 'WhatsApp niet verbonden' });

  const { contact, message } = req.body;
  if (!contact || !message) {
    return res.status(400).json({ error: 'contact en message zijn verplicht' });
  }

  try {
    const contacts = await client.getContacts();
    const query = contact.toLowerCase();
    const match = contacts.find(
      (c) => c.name && c.name.toLowerCase().includes(query)
    );

    if (!match) {
      return res.status(404).json({ error: `Contact "${contact}" niet gevonden` });
    }

    await client.sendMessage(match.id._serialized, message);
    res.json({ status: 'sent', contact: match.name, number: match.id.user });
  } catch (e) {
    console.error('Send failed:', e);
    res.status(500).json({ error: e.message });
  }
});

// List contacts
app.get('/contacts', async (_req, res) => {
  if (!ready) return res.status(503).json({ error: 'WhatsApp niet verbonden' });

  try {
    const contacts = await client.getContacts();
    const list = contacts
      .filter((c) => c.isMyContact && c.name)
      .map((c) => ({ name: c.name, number: c.id.user }))
      .sort((a, b) => a.name.localeCompare(b.name));
    res.json(list);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Initialize default client (QR mode)
client = createClient();
wireEvents(client);
client.initialize().catch((err) => {
  console.error('WhatsApp initialization failed:', err);
  lastError = err.message;
});

app.listen(PORT, () => {
  console.log(`WhatsApp service listening on :${PORT}`);
});
