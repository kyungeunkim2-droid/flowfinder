import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

// FlowFinder Nano Banana preview server
// 1) npm install
// 2) copy .env.example to .env and set GEMINI_API_KEY
// 3) npm run dev

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
const PORT = process.env.PORT || 3000;
const MODEL = process.env.GEMINI_IMAGE_MODEL || 'gemini-2.0-flash-preview-image-generation';

app.use(express.json({ limit: '25mb' }));
app.use(express.static(__dirname, {
  etag: true,
  maxAge: '1h',
}));

app.get('/api/health', (req, res) => {
  res.json({ ok: true, hasGeminiKey: Boolean(process.env.GEMINI_API_KEY), model: MODEL });
});

function assertSafeUrl(raw, baseUrl) {
  if (!raw || typeof raw !== 'string') return null;
  const url = raw.trim();
  if (url.startsWith('data:')) return null;
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  return new URL(url, baseUrl).toString();
}

async function loadImagePart(src, label, baseUrl) {
  const safeUrl = assertSafeUrl(src, baseUrl);
  if (!safeUrl) return null;
  const response = await fetch(safeUrl);
  if (!response.ok) throw new Error(`이미지 로드 실패: ${label || src}`);
  const mimeType = response.headers.get('content-type')?.split(';')[0] || 'image/png';
  const bytes = Buffer.from(await response.arrayBuffer());
  return {
    inlineData: {
      mimeType,
      data: bytes.toString('base64'),
    },
  };
}

app.post('/api/generate-preview', async (req, res) => {
  console.log('[NanoBanana] /api/generate-preview called');
  try {
    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({
        error: 'GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.',
      });
    }

    const baseUrl = `${req.protocol}://${req.get('host')}/`;

    const {
      deskImage,
      screenImage,
      topTexture,
      legTexture,
      screenTexture,
      topCode,
      legCode,
      screenCode,
      deskLabel,
      screenLabel,
      legType,
      casterType,
      topShape,
      size,
    } = req.body || {};

    const parts = [];
    parts.push({
      text: [
        'You are performing strict product-photo editing for a premium office furniture configurator.',
        'Use the provided official product photo as the exact base image and keep the same camera angle, silhouette, dimensions, proportions, edges, screws, pads, wheels, screen position, desk leg position, shadows, highlights, and white/transparent background.',
        'Do not generate a new desk. Do not redraw the product. Do not change the model design, perspective, size, composition, number of legs, or background.',
        `Desk product: ${deskLabel || 'selected desk'}.`,
        screenLabel ? `Screen product: ${screenLabel}.` : 'No screen product selected unless a screen image is provided.',
        topCode ? `Only replace the visible desktop surface finish with the provided top material reference: ${topCode}.` : 'Keep the desktop surface as-is if no top material is provided.',
        legCode ? `Only replace the visible desk leg finish with the provided leg material reference: ${legCode}.` : 'Keep the legs as-is if no leg material is provided.',
        screenCode ? `Only replace the visible screen panel finish with the provided screen material reference: ${screenCode}.` : 'Keep the screen as-is if no screen material is provided.',
        legType ? `The desk leg shape must remain or be edited to match this selected option: ${legType}.` : 'Keep the original leg shape if no leg shape option is provided.',
        casterType ? `The bottom support option must match this selected option: ${casterType}. If caster/wheels are selected, show wheels; if glide is selected, show glides.` : 'Keep the original bottom support if no caster/glide option is provided.',
        topShape ? `The desktop corner/edge shape must match this selected option: ${topShape}. If round is selected, make the top corners rounded; if square/straight is selected, keep square corners.` : 'Keep the original desktop corner shape if no top shape option is provided.',
        size && (size.w || size.d || size.h) ? `Preserve the selected approximate size proportions: W ${size.w || 'default'}mm, D ${size.d || 'default'}mm, H ${size.h || 'default'}mm.` : 'Preserve the original size proportions.',
        'Match material reference images closely, including color, grain, fabric, PET, or metal finish, but keep all original lighting and shadows.',
        'The final image must look like the same official product photo with only material finishes changed. No labels, no extra furniture, no watermark.',
      ].join(' '),
    });

    const imageInputs = [
      ['desk product image', deskImage],
      ['screen product image', screenImage],
      ['desktop material reference', topTexture],
      ['leg material reference', legTexture],
      ['screen material reference', screenTexture],
    ];

    for (const [label, src] of imageInputs) {
      const part = await loadImagePart(src, label, baseUrl).catch((err) => {
        console.warn(err.message);
        return null;
      });
      if (part) {
        parts.push({ text: `Reference image: ${label}.` });
        parts.push(part);
      }
    }

    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    const response = await ai.models.generateContent({
      model: MODEL,
      contents: [{ role: 'user', parts }],
    });

    const outPart = response?.candidates?.[0]?.content?.parts?.find((part) => part.inlineData || part.inline_data);
    const inline = outPart?.inlineData || outPart?.inline_data;
    if (!inline?.data) {
      const text = response?.candidates?.[0]?.content?.parts?.map((part) => part.text).filter(Boolean).join('\n') || '';
      return res.status(502).json({ error: '이미지 결과를 받지 못했습니다.', detail: text });
    }

    const mimeType = inline.mimeType || inline.mime_type || 'image/png';
    res.json({ imageUrl: `data:${mimeType};base64,${inline.data}` });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message || '이미지 생성 실패' });
  }
});

app.listen(PORT, () => {
  console.log(`FlowFinder preview server running: http://localhost:${PORT}`);
});
