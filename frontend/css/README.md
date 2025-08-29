# CSS Design System

This directory contains the comprehensive CSS design system for the XAI Kreditprüfung application, featuring modern design elements, glassmorphism effects, and professional styling.

## File Structure

### Core CSS Files

- **`main.css`** - Main CSS file containing:
  - CSS variables and root styles
  - Global styles and layout utilities
  - Typography system
  - Spacing utilities
  - Glassmorphism components
  - Button styles
  - Form elements
  - Card components
  - Streamlit-specific overrides

- **`components.css`** - Component-specific styles including:
  - Hero sections
  - Feature cards
  - Stats sections
  - Form sections
  - Result cards
  - Chart containers
  - Loading states
  - Tooltips
  - Modals
  - Breadcrumb navigation
  - Progress indicators
  - Status indicators

- **`animations.css`** - Animation and effects including:
  - Keyframe animations (fade, slide, scale, rotate, bounce)
  - Animation classes
  - Animation delays and durations
  - Hover effects
  - Interactive effects
  - Loading animations
  - Scroll animations
  - Parallax effects
  - Text animations
  - Performance optimizations

- **`sidebar.css`** - Sidebar-specific styling including:
  - Sidebar container styles
  - Navigation items
  - Active states
  - Hover effects
  - Responsive design
  - Animations
  - Scrollbar styling

## Design Features

### 🎨 Color Palette
- **Primary**: `#C24914` (deep burnt orange)
- **Secondary**: `#F4A261` (soft amber)
- **Accent**: `#E76F51` (coral)
- **Background**: `#0B0E14` (very dark backdrop)
- **Surface**: Various transparency levels for glassmorphism

### 🌟 Glassmorphism Effects
- Backdrop blur effects
- Semi-transparent backgrounds
- Subtle borders
- Layered shadows
- Hover transformations

### ✨ Animations
- Smooth transitions (0.3s cubic-bezier)
- Hover effects with lift and glow
- Loading animations
- Scroll-triggered animations
- Micro-interactions

### 📱 Responsive Design
- Mobile-first approach
- Flexible grid systems
- Adaptive typography
- Touch-friendly interactions

## Usage Examples

### Basic Glass Card
```html
<div class="glass-card">
  <h3>Card Title</h3>
  <p>Card content goes here</p>
</div>
```

### Animated Component
```html
<div class="feature-card animate-fade-in-up delay-300">
  <div class="feature-icon">🚀</div>
  <h3 class="feature-title">Feature</h3>
  <p class="feature-description">Description</p>
</div>
```

### Button with Hover Effects
```html
<button class="btn btn-primary hover-lift">
  Click Me
</button>
```

### Form Section
```html
<div class="form-section">
  <div class="form-section-header">
    <div class="form-section-icon">💰</div>
    <h3 class="form-section-title">Financial Information</h3>
  </div>
  <!-- Form content -->
</div>
```

## CSS Classes Reference

### Layout Classes
- `.container` - Main container with max-width
- `.container-wide` - Wide container variant
- `.glass` - Basic glassmorphism effect
- `.glass-card` - Card with glassmorphism
- `.glass-panel` - Panel with enhanced glassmorphism

### Typography Classes
- `.text-display` - Display font family
- `.text-heading` - Heading font family
- `.text-body` - Body text
- `.text-caption` - Caption text
- `.text-gradient` - Gradient text effect
- `.text-muted` - Muted text color

### Spacing Classes
- `.m-{1-6}` - Margin utilities
- `.p-{1-6}` - Padding utilities
- `.mt-{1-6}` - Margin top
- `.mb-{1-6}` - Margin bottom
- `.pt-{1-6}` - Padding top
- `.pb-{1-6}` - Padding bottom

### Animation Classes
- `.animate-fade-in` - Fade in animation
- `.animate-fade-in-up` - Fade in from bottom
- `.animate-slide-in-left` - Slide in from left
- `.delay-{100-1000}` - Animation delays
- `.duration-{100-1000}` - Animation durations

### Hover Effects
- `.hover-lift` - Lift on hover
- `.hover-scale` - Scale on hover
- `.hover-glow` - Glow effect on hover
- `.hover-border-glow` - Border glow on hover

### Component Classes
- `.feature-card` - Feature card component
- `.stat-card` - Statistics card
- `.result-card` - Result display card
- `.chart-container` - Chart wrapper
- `.loading-container` - Loading state container
- `.alert` - Alert component
- `.badge` - Badge component

## Browser Support

- Modern browsers with CSS Grid and Flexbox support
- Backdrop-filter support for glassmorphism effects
- CSS custom properties (CSS variables)
- CSS animations and transitions

## Performance Considerations

- GPU-accelerated animations using `transform` and `opacity`
- Efficient hover effects with `will-change` property
- Reduced motion support for accessibility
- Optimized transitions and animations

## Accessibility Features

- High contrast color combinations
- Focus states for keyboard navigation
- Reduced motion support
- Semantic HTML structure
- Screen reader friendly

## Customization

The design system is built with CSS custom properties, making it easy to customize:

```css
:root {
  --color-primary: #your-color;
  --spacing-md: 20px;
  --radius-lg: 20px;
  --transition-normal: 0.4s ease;
}
```

## Maintenance

- Keep CSS files modular and organized
- Use consistent naming conventions
- Document new components and utilities
- Test across different screen sizes
- Validate CSS for errors
- Optimize for performance
