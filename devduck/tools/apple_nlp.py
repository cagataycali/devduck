"""🧠 Apple NLP - On-device language detection, embeddings, sentiment via Neural Engine."""

from strands import tool
from typing import Dict, Any, List


@tool
def apple_nlp(
    action: str = "detect",
    text: str = "",
    word: str = None,
    word2: str = None,
    language: str = "en",
    top_k: int = 5,
) -> Dict[str, Any]:
    """🧠 On-device NLP via Apple NaturalLanguage framework. Zero cloud, instant.

    Uses Apple's Neural Engine for:
    - Language detection (50+ languages)
    - Word embeddings (300-dim vectors, built into macOS)
    - Semantic similarity (word/sentence comparison)
    - Named entity recognition
    - Part-of-speech tagging
    - Tokenization / lemmatization
    - Sentiment analysis

    Args:
        action: Action to perform:
            - "detect": Detect language of text
            - "embed": Get word embedding vector
            - "similar": Find similar words (nearest neighbors)
            - "distance": Semantic distance between two words
            - "entities": Named entity recognition
            - "pos": Part-of-speech tagging
            - "tokenize": Tokenize text
            - "lemma": Lemmatize text
            - "sentiment": Sentiment analysis
            - "tag": Full linguistic tagging
        text: Input text to analyze
        word: Word for embedding operations
        word2: Second word for distance comparison
        language: Language code (default: 'en')
        top_k: Number of similar words to return

    Returns:
        Dict with NLP results
    """
    try:
        from NaturalLanguage import (
            NLLanguageRecognizer,
            NLEmbedding,
            NLTagger,
            NLTokenizer,
            NLTagSchemeLanguage,
            NLTagSchemeTokenType,
            NLTagSchemeLexicalClass,
            NLTagSchemeNameType,
            NLTagSchemeLemma,
            NLTagSchemeSentimentScore,
            NLTagSchemeNameTypeOrLexicalClass,
        )
        from Foundation import NSRange, NSMakeRange

        if action == "detect":
            if not text:
                return {"status": "error", "content": [{"text": "text parameter required"}]}

            recognizer = NLLanguageRecognizer.alloc().init()
            recognizer.processString_(text)

            dominant = recognizer.dominantLanguage()
            hypotheses = recognizer.languageHypothesesWithMaximum_(5)

            lines = [f"🌍 Language Detection:"]
            lines.append(f"  Dominant: {dominant}")
            lines.append(f"\n  Hypotheses:")
            for lang, prob in sorted(hypotheses.items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 20)
                lines.append(f"    {lang:8} {prob:6.1%} {bar}")

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "embed":
            word_to_embed = word or (text.split()[0] if text else None)
            if not word_to_embed:
                return {"status": "error", "content": [{"text": "word or text required"}]}

            embedding = NLEmbedding.wordEmbeddingForLanguage_(language)
            if not embedding:
                return {"status": "error", "content": [{"text": f"No embedding model for language '{language}'"}]}

            vec = embedding.vectorForString_(word_to_embed)
            if not vec:
                return {"status": "success", "content": [{"text": f"Word '{word_to_embed}' not in vocabulary"}]}

            vec_list = list(vec)
            preview = [f"{v:.4f}" for v in vec_list[:10]]

            text_out = f"🧠 Embedding for '{word_to_embed}' ({language}):\n"
            text_out += f"  Dimensions: {len(vec_list)}\n"
            text_out += f"  Vector (first 10): [{', '.join(preview)}...]\n"
            text_out += f"  Norm: {sum(v*v for v in vec_list)**0.5:.4f}"

            return {"status": "success", "content": [{"text": text_out}]}

        elif action == "similar":
            target = word or (text.split()[0] if text else None)
            if not target:
                return {"status": "error", "content": [{"text": "word or text required"}]}

            embedding = NLEmbedding.wordEmbeddingForLanguage_(language)
            if not embedding:
                return {"status": "error", "content": [{"text": f"No embedding model for '{language}'"}]}

            # API returns array of strings, compute distances separately
            neighbors = embedding.neighborsForString_maximumCount_distanceType_(target, top_k, 0)
            if not neighbors:
                return {"status": "success", "content": [{"text": f"No neighbors found for '{target}'"}]}

            lines = [f"🔍 Most similar to '{target}' ({language}):\n"]
            for neighbor_word in neighbors:
                distance = embedding.distanceBetweenString_andString_distanceType_(target, neighbor_word, 0)
                similarity = max(0, 1 - distance)
                bar = "█" * int(similarity * 20)
                lines.append(f"  {str(neighbor_word):20} dist={distance:.4f} sim={similarity:.1%} {bar}")

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "distance":
            w1 = word or (text.split()[0] if text and len(text.split()) >= 1 else None)
            w2 = word2 or (text.split()[1] if text and len(text.split()) >= 2 else None)
            if not w1 or not w2:
                return {"status": "error", "content": [{"text": "Need word + word2, or text with two words"}]}

            embedding = NLEmbedding.wordEmbeddingForLanguage_(language)
            if not embedding:
                return {"status": "error", "content": [{"text": f"No embedding for '{language}'"}]}

            dist = embedding.distanceBetweenString_andString_distanceType_(w1, w2, 0)
            similarity = max(0, 1 - dist)

            text_out = f"📏 Semantic Distance:\n"
            text_out += f"  '{w1}' ↔ '{w2}'\n"
            text_out += f"  Distance: {dist:.4f}\n"
            text_out += f"  Similarity: {similarity:.1%}\n"
            text_out += f"  {'█' * int(similarity * 30)}{'░' * (30 - int(similarity * 30))}"

            return {"status": "success", "content": [{"text": text_out}]}

        elif action == "entities":
            if not text:
                return {"status": "error", "content": [{"text": "text required"}]}

            tagger = NLTagger.alloc().initWithTagSchemes_([NLTagSchemeNameType])
            tagger.setString_(text)

            entities = []
            text_range = NSMakeRange(0, len(text))

            def handler(tag, token_range, stop):
                if tag:
                    token = text[token_range.location:token_range.location + token_range.length]
                    entities.append({"text": token, "type": str(tag), "pos": token_range.location})

            tagger.enumerateTagsInRange_unit_scheme_options_usingBlock_(
                text_range, 0, NLTagSchemeNameType, 0, handler
            )

            if not entities:
                return {"status": "success", "content": [{"text": "No named entities found."}]}

            lines = [f"🏷️ Named Entities ({len(entities)}):\n"]
            for e in entities:
                emoji = {"PersonalName": "👤", "PlaceName": "📍", "OrganizationName": "🏢"}.get(e["type"], "🔖")
                lines.append(f"  {emoji} {e['text']:30} [{e['type']}]")

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "pos":
            if not text:
                return {"status": "error", "content": [{"text": "text required"}]}

            tagger = NLTagger.alloc().initWithTagSchemes_([NLTagSchemeLexicalClass])
            tagger.setString_(text)

            tags = []
            text_range = NSMakeRange(0, len(text))

            def handler(tag, token_range, stop):
                if tag:
                    token = text[token_range.location:token_range.location + token_range.length]
                    tags.append({"text": token.strip(), "tag": str(tag)})

            tagger.enumerateTagsInRange_unit_scheme_options_usingBlock_(
                text_range, 0, NLTagSchemeLexicalClass, 0, handler
            )

            lines = [f"📝 Part-of-Speech Tags:\n"]
            for t in tags:
                if t["text"]:
                    lines.append(f"  {t['text']:20} → {t['tag']}")

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "lemma":
            if not text:
                return {"status": "error", "content": [{"text": "text required"}]}

            tagger = NLTagger.alloc().initWithTagSchemes_([NLTagSchemeLemma])
            tagger.setString_(text)

            lemmas = []
            text_range = NSMakeRange(0, len(text))

            def handler(tag, token_range, stop):
                token = text[token_range.location:token_range.location + token_range.length]
                if tag and token.strip():
                    lemmas.append({"word": token.strip(), "lemma": str(tag)})

            tagger.enumerateTagsInRange_unit_scheme_options_usingBlock_(
                text_range, 0, NLTagSchemeLemma, 0, handler
            )

            lines = [f"📖 Lemmatization:\n"]
            for l in lemmas:
                changed = " ✨" if l["word"].lower() != l["lemma"].lower() else ""
                lines.append(f"  {l['word']:20} → {l['lemma']}{changed}")

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "sentiment":
            if not text:
                return {"status": "error", "content": [{"text": "text required"}]}

            tagger = NLTagger.alloc().initWithTagSchemes_([NLTagSchemeSentimentScore])
            tagger.setString_(text)

            text_range = NSMakeRange(0, len(text))

            # Get per-sentence sentiment
            sentences = []
            def handler(tag, token_range, stop):
                if tag is not None:
                    sent = text[token_range.location:token_range.location + token_range.length]
                    try:
                        score = float(str(tag))
                    except (ValueError, TypeError):
                        score = 0.0
                    sentences.append({"text": sent.strip(), "score": score})

            tagger.enumerateTagsInRange_unit_scheme_options_usingBlock_(
                text_range, 1, NLTagSchemeSentimentScore, 0, handler
            )

            # Compute overall from sentences
            if sentences:
                overall = sum(s["score"] for s in sentences) / len(sentences)
            else:
                overall = 0.0

            if overall > 0.1:
                mood = "😊 Positive"
            elif overall < -0.1:
                mood = "😞 Negative"
            else:
                mood = "😐 Neutral"

            lines = [f"💭 Sentiment Analysis:\n  Overall: {mood} (score: {overall:+.3f})\n"]
            if sentences:
                lines.append("  Per-sentence:")
                for s in sentences:
                    emoji = "😊" if s["score"] > 0.1 else ("😞" if s["score"] < -0.1 else "😐")
                    lines.append(f"    {emoji} [{s['score']:+.3f}] {s['text'][:80]}")

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "tokenize":
            if not text:
                return {"status": "error", "content": [{"text": "text required"}]}

            tokenizer = NLTokenizer.alloc().initWithUnit_(0)  # 0 = word
            tokenizer.setString_(text)
            text_range = NSMakeRange(0, len(text))

            tokens = []
            token_range = tokenizer.tokenRangeAtIndex_(0)
            while token_range.location < len(text):
                token = text[token_range.location:token_range.location + token_range.length]
                tokens.append(token)
                next_start = token_range.location + token_range.length
                if next_start >= len(text):
                    break
                token_range = tokenizer.tokenRangeAtIndex_(next_start)
                if token_range.location <= tokens[-1] if isinstance(tokens[-1], int) else token_range.location < next_start:
                    break

            return {"status": "success", "content": [{"text": f"🔤 Tokens ({len(tokens)}): {tokens}"}]}

        else:
            return {"status": "error", "content": [{"text": f"Unknown action: {action}. Use: detect, embed, similar, distance, entities, pos, lemma, sentiment, tokenize"}]}

    except ImportError:
        return {"status": "error", "content": [{"text": "Install: pip install pyobjc-framework-NaturalLanguage"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}
