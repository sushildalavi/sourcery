import { ChatShell } from '../components/chat/ChatShell';

const PROMPTS = [
  {
    eyebrow: 'Summary',
    title: 'Summarize this document',
    description: 'Get the main findings, structure, and key takeaways.',
    prompt: 'Summarize the selected uploaded document.',
  },
  {
    eyebrow: 'Key points',
    title: 'Extract key concepts',
    description: 'Pull out the strongest claims, topics, and themes.',
    prompt: 'Extract the key skills, technical topics, and standout claims from the selected uploaded document.',
  },
  {
    eyebrow: 'Evidence',
    title: 'Inspect evidence',
    description: 'Surface the passages that best support the main claims.',
    prompt: 'What evidence best supports the main claims in this document?',
  },
  {
    eyebrow: 'Gaps',
    title: 'Find gaps and risks',
    description: 'Identify weak support, ambiguities, or missing details.',
    prompt: 'Identify weakly supported claims or missing details in this document.',
  },
];

export default function UploadedChat() {
  return (
    <ChatShell
      mode="uploaded"
      title="Document-grounded chat"
      description="Upload a PDF from the panel, select it, and ask grounded questions. Every answer is backed by specific passages."
      emptyPrompts={PROMPTS}
    />
  );
}
