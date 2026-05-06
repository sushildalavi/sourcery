import { ChatShell } from '../components/chat/ChatShell';

const PROMPTS = [
  {
    eyebrow: 'Explain',
    title: 'Understand a concept',
    description: 'Get a clear, cited explanation of any research topic.',
    prompt: 'Explain how attention mechanisms work in transformers.',
  },
  {
    eyebrow: 'Discover',
    title: 'Find relevant papers',
    description: 'Surface the strongest papers, surveys, and references.',
    prompt: 'Find recent papers on retrieval-augmented generation.',
  },
  {
    eyebrow: 'Compare',
    title: 'Compare approaches',
    description: 'Synthesize tradeoffs across methods with source evidence.',
    prompt: 'Compare BERT, GPT, and T5 architectures.',
  },
  {
    eyebrow: 'Survey',
    title: 'Literature overview',
    description: 'Get a structured overview of a research area with key works.',
    prompt: 'Give me a literature overview of graph neural networks.',
  },
];

export default function PublicChat() {
  return (
    <ChatShell
      mode="public"
      title="Public literature chat"
      description="Search across Semantic Scholar, OpenAlex, arXiv, and Crossref. Get grounded, cited answers from millions of papers."
      emptyPrompts={PROMPTS}
    />
  );
}
