import { useEffect, useRef, useState } from "react";

const imagePaths = {
  earth: "/images/earth-hero.webp",
  coast: "/images/coastline.webp",
  farm: "/images/farmland.webp",
  city: "/images/urban.webp",
  quantum: "/images/quantum-hardware.webp",
};

function Visual({ name, className = "", children }) {
  const [loaded, setLoaded] = useState(false);
  return (
    <div className={`visual visual--${name} ${className}`}>
      <div className="visual-art" aria-hidden="true" />
      <img src={imagePaths[name]} alt="" loading={name === "earth" ? "eager" : "lazy"} onLoad={() => setLoaded(true)} onError={() => setLoaded(false)} className={loaded ? "visual-photo is-loaded" : "visual-photo"} />
      {children}
    </div>
  );
}

function Arrow({ diagonal = false }) {
  return <span aria-hidden="true">{diagonal ? "↗" : "→"}</span>;
}

function SectionHeading({ eyebrow, title, body, children }) {
  return (
    <div className="section-heading reveal">
      <p className="eyebrow"><span className="eyebrow-mark" />{eyebrow}</p>
      <h2>{title}</h2>
      {body && <p className="section-description">{body}</p>}
      {children}
    </div>
  );
}

function ModelCard({ type }) {
  const quantum = type === "quantum";
  return (
    <article className={`model-card ${quantum ? "model-card--quantum" : ""}`}>
      <div className="model-card-top">
        <span className="model-glyph">{quantum ? "✳" : "◉"}</span>
        <span className="model-index">{quantum ? "MODEL 02" : "MODEL 01"}</span>
      </div>
      <h3>{quantum ? "Hybrid quantum" : "Classical AI"}</h3>
      <p>{quantum ? "A conventional feature extractor passes compact image clues to a small quantum circuit." : "A conventional neural model learns visual patterns from satellite images."}</p>
      <div className="model-flow">
        <span>IMAGE</span><Arrow /><span>{quantum ? "FEATURES" : "CNN"}</span><Arrow /><span>{quantum ? "QNN" : "CLASSIFIER"}</span>
      </div>
      <div className="model-card-footer"><span className="status-dot" /> Not connected <span>Baseline {quantum ? "B" : "A"}</span></div>
    </article>
  );
}

function Prototype() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function receiveFile(candidate) {
    if (!candidate) return;
    if (!["image/png", "image/jpeg", "image/webp"].includes(candidate.type)) {
      setError("Choose a PNG, JPEG or WebP image.");
      return;
    }
    if (candidate.size > 10 * 1024 * 1024) {
      setError("Image must be 10 MB or smaller.");
      return;
    }
    setFile(candidate);
    setError("");
  }

  return (
    <div className="lab-grid">
      <div
        className={`upload-area ${dragging ? "is-dragging" : ""} ${preview ? "has-preview" : ""}`}
        onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
        onDragLeave={(event) => { event.preventDefault(); setDragging(false); }}
        onDrop={(event) => { event.preventDefault(); setDragging(false); receiveFile(event.dataTransfer.files?.[0]); }}
      >
        {preview ? <img className="upload-preview" src={preview} alt="Your uploaded satellite image preview" /> : <div className="upload-orbit" aria-hidden="true"><span>＋</span></div>}
        <div className="upload-copy">
          <p className="upload-kicker">{preview ? "IMAGE READY" : "INPUT / SATELLITE IMAGERY"}</p>
          <h3>{preview ? file.name : "Drop an image of Earth."}</h3>
          <p>{preview ? "Preview stays on this device. No model has been run." : "Upload a satellite image to preview it in the lab."}</p>
          <button type="button" className="button button--light" onClick={() => inputRef.current?.click()}>{preview ? "Change image" : "Choose image"} <Arrow diagonal /></button>
          <input ref={inputRef} type="file" accept="image/png,image/jpeg,image/webp" aria-label="Choose a satellite image" hidden onChange={(event) => { receiveFile(event.target.files?.[0]); event.target.value = ""; }} />
          {error && <p className="file-error" role="alert">{error}</p>}
          <p className="upload-note">PNG, JPEG, WebP · up to 10 MB</p>
        </div>
      </div>
      <div className="results-area">
        <div className="results-top"><span>OUTPUT / MODEL COMPARISON</span><span className="live-tag">OFFLINE PROTOTYPE</span></div>
        <div className="result-row">
          <div className="result-symbol result-symbol--classical">◉</div>
          <div><p>CLASSICAL MODEL</p><strong>Not connected</strong><small>Prediction will appear after model integration.</small></div>
          <span className="result-arrow">↗</span>
        </div>
        <div className="result-row">
          <div className="result-symbol result-symbol--quantum">✳</div>
          <div><p>QUANTUM MODEL</p><strong>Not connected</strong><small>Prediction will appear after model integration.</small></div>
          <span className="result-arrow">↗</span>
        </div>
        <p className="lab-disclaimer">No predictions, probabilities, accuracy values or processing times are fabricated in this interface. Real models will use a shared evaluation dataset and documented preprocessing.</p>
      </div>
    </div>
  );
}

export default function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    }, { threshold: 0.12 });
    document.querySelectorAll(".reveal").forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, []);
  const closeMenu = () => setMenuOpen(false);

  return (
    <>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Quantum Earth home" onClick={closeMenu}><span className="brand-icon">✳</span><span>QUANTUM<span className="brand-muted">EARTH</span></span></a>
        <button type="button" className="menu-toggle" aria-expanded={menuOpen} aria-controls="primary-navigation" onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? "Close" : "Menu"} <span>{menuOpen ? "×" : "☰"}</span></button>
        <nav id="primary-navigation" className={menuOpen ? "nav is-open" : "nav"} aria-label="Main navigation">
          <a href="#mission" onClick={closeMenu}>Mission</a><a href="#approach" onClick={closeMenu}>Approach</a><a href="#compare" onClick={closeMenu}>Compare</a>
          <a className="nav-lab" href="#lab" onClick={closeMenu}>Explore lab <Arrow diagonal /></a>
        </nav>
      </header>

      <main>
        <section id="top" className="hero">
          <div className="hero-grid" aria-hidden="true" />
          <Visual name="earth" className="hero-visual" />
          <div className="hero-shade" />
          <div className="hero-content">
            <p className="eyebrow hero-eyebrow"><span className="eyebrow-mark" />EARTH OBSERVATION / EXPERIMENT 001</p>
            <h1>See Earth.<br /><em>Think beyond.</em></h1>
            <p className="hero-copy">One planet. Two ways to understand it. Exploring classical and quantum machine learning through satellite imagery.</p>
            <a className="button button--outline" href="#mission">Explore the mission <span>↓</span></a>
          </div>
          <div className="hero-bottom"><span>01 / 06 · QUANTUM EARTH</span><span>SCROLL TO EXPLORE ↓</span><span>PROTOTYPE V1</span></div>
        </section>

        <section id="mission" className="section mission">
          <div className="container mission-layout">
            <SectionHeading eyebrow="01 / THE MISSION" title={<>The planet is speaking.<br /><em>Can AI understand it?</em></>} body="Every satellite image carries signals of water, vegetation, cities and change. Our experiment investigates how two different model families interpret those signals." />
            <div className="mission-visual reveal">
              <Visual name="coast" className="mission-photo"><span className="image-label image-label--top">OBSERVATION / 001</span><span className="image-label image-label--bottom">LAND · WATER · TEXTURE</span></Visual>
              <span className="corner-decoration" aria-hidden="true">+</span>
            </div>
          </div>
          <div className="container data-strip reveal"><span>ONE IMAGE</span><span>↗</span><span>MANY SIGNALS</span><span>↗</span><span>TWO APPROACHES</span></div>
        </section>

        <section className="section landscapes">
          <div className="container">
            <SectionHeading eyebrow="02 / THE SIGNALS" title={<>Different landscapes.<br /><em>Different clues.</em></>} body="Color, shape and texture help models distinguish what is happening on the ground." />
            <div className="landscape-grid">
              <article className="landscape-card reveal"><Visual name="farm"><span className="image-label">01 / AGRICULTURE</span></Visual><div><span>FIELD PATTERNS</span><Arrow diagonal /></div></article>
              <article className="landscape-card landscape-card--offset reveal"><Visual name="city"><span className="image-label">02 / URBAN</span></Visual><div><span>HUMAN STRUCTURES</span><Arrow diagonal /></div></article>
            </div>
          </div>
        </section>

        <section id="approach" className="section approach">
          <div className="container">
            <SectionHeading eyebrow="03 / TWO APPROACHES" title={<>Same Earth.<br /><em>Different computation.</em></>} body="The experiment starts with two separate baselines, trained and evaluated fairly on the same dataset." />
            <div className="model-grid reveal"><ModelCard type="classical" /><ModelCard type="quantum" /></div>
          </div>
          <div className="quantum-feature container reveal">
            <Visual name="quantum" className="quantum-art" />
            <div className="quantum-feature-copy"><p className="eyebrow"><span className="eyebrow-mark" />INSIDE THE HYBRID MODEL</p><h3>A new way to connect the clues.</h3><p>A classical encoder extracts compact information from an image. A small quantum circuit then processes those features before the classifier produces a result.</p><p className="feature-note">A research question, not a claim of quantum advantage.</p></div>
          </div>
        </section>

        <section id="compare" className="section compare">
          <div className="container">
            <SectionHeading eyebrow="04 / THE COMPARISON" title={<>Not a race for hype.<br /><em>A test for evidence.</em></>} body="A meaningful comparison controls the dataset and evaluation conditions. A quantum model earns its place through measured results, not assumptions." />
            <div className="comparison-table reveal" role="table" aria-label="Planned model comparison">
              <div className="comparison-row comparison-row--head" role="row"><span role="columnheader">MEASURE</span><span role="columnheader">CLASSICAL</span><span role="columnheader">QUANTUM</span></div>
              <div className="comparison-row" role="row"><span role="cell">Prediction</span><span role="cell">Pending model</span><span role="cell">Pending model</span></div>
              <div className="comparison-row" role="row"><span role="cell">Accuracy</span><span role="cell">Not measured</span><span role="cell">Not measured</span></div>
              <div className="comparison-row" role="row"><span role="cell">Inference time</span><span role="cell">Not measured</span><span role="cell">Not measured</span></div>
              <div className="comparison-row" role="row"><span role="cell">Shared test set</span><span role="cell">Planned</span><span role="cell">Planned</span></div>
            </div>
          </div>
        </section>

        <section id="lab" className="section lab">
          <div className="container">
            <SectionHeading eyebrow="05 / INTERACTIVE LAB" title={<>Your view from orbit.<br /><em>Our next experiment.</em></>} body="Try the image preview now. Real inference will become available once both baselines have been implemented and validated." />
            <Prototype />
          </div>
        </section>

        <section className="closing">
          <div className="closing-orbit" aria-hidden="true" />
          <div className="container closing-content reveal"><p className="eyebrow"><span className="eyebrow-mark" />06 / WHAT COMES NEXT</p><h2>From observation<br />to <em>understanding.</em></h2><p>Next: develop and evaluate a structured quantum feature model for the relationships between satellite-image clues.</p><a href="#top" className="button button--outline">Back to Earth ↑</a></div>
          <footer className="container footer"><span className="brand"><span className="brand-icon">✳</span>QUANTUM<span className="brand-muted">EARTH</span></span><span>RESEARCH PROTOTYPE / V1</span><a href="https://github.com/tej-1916/quantum-earth" target="_blank" rel="noreferrer">GITHUB <Arrow diagonal /></a></footer>
        </section>
      </main>
    </>
  );
}
