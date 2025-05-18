import React, { useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import '../../assets/css/style.css'
import '../../assets/css/light.css'
// import '../../assets/css/dark.css'
import '../../assets/css/plugins.min.css'

const BannerCreator = () => {
    const location = useLocation();

    useEffect(() => {
        // Save original overflow style
        document.body.classList.add("banner-page");

        return () => {
            // Remove the class when the component unmounts
            document.body.classList.remove("banner-page");
        };
    }, []);
    // Parse the query parameters
    const queryParams = new URLSearchParams(location.search);
    const billboardId = queryParams.get("billboardId");

    return (
        <>
            <main id="pmotion" draggable="false" className="loading">
                {/* <!-- Page Loader START --> */}
                <div id="main-loader" className="loader-wrap">
                    <div className="loader-inner">
                        <div className="loader"></div>
                    </div>
                </div>
                {/* <!-- Page Loader END --> */}
                {/* <!-- TOOLBAR --> */}
                <div id="toolbar" className="noselect">
                    <div id="logo">
                        <a href="#">
                            <img src="logo.jpeg" alt="Palleon Motion" />
                        </a>
                    </div>
                    <div id="tool-wrap">
                        <div id="object-settings-select" className="tool tool-active" data-id="object-settings">
                            <span className="material-icons">tune</span>
                            <p>Settings</p>
                        </div>
                        <div id="project-tool-select" className="tool" data-id="project-tool">
                            <span className="material-icons">perm_media</span>
                            <p>Templates</p>
                        </div>
                        <div id="upload-tool-select" className="tool" data-id="upload-tool">
                            <span className="material-icons">cloud_upload</span>
                            <p>Uploads</p>
                        </div>
                        <div id="text-tool-select" className="tool" data-id="text-tool">
                            <span className="material-icons">title</span>
                            <p>Text</p>
                        </div>
                        <div id="shape-tool-select" className="tool" data-id="shape-tool">
                            <span className="material-icons">interests</span>
                            <p>Objects</p>
                        </div>
                        <div id="image-tool-select" className="tool" data-id="image-tool">
                            <span className="material-icons">add_photo_alternate</span>
                            <p>Images</p>
                        </div>
                        {/* <div id="video-tool-select" className="tool" data-id="video-tool">
                            <span className="material-icons">videocam</span>
                            <p>Videos</p>
                        </div>
                        <div id="audio-tool-select" className="tool" data-id="audio-tool">
                            <span className="material-icons">queue_music</span>
                            <p>Audio</p>
                        </div>
                        <div id="qrcode-tool-select" className="tool" data-id="qrcode-tool">
                            <span className="material-icons">qr_code</span>
                            <p>QR CODE</p>
                        </div> */}
                    </div>
                </div>
                {/* <!-- TOOLBAR END -->
                <!-- BROWSER --> */}
                <div id="browser" className="scrollingPanel">
                    <div id="browser-container">
                        {/* <!-- OBJECT SETTINGS --> */}
                        <div id="object-settings">
                            <div className="tool-content">
                                <p className="property-title">Settings</p>
                                <span className="collapse material-icons">
                                    keyboard_double_arrow_left
                                </span>
                                <div id="properties">
                                    <div id="properties-overlay">
                                        <button id="crop-selection" type="button" className="btn btn-full primary"> <span className="material-icons">transform</span>Click To Crop Image</button>
                                    </div>
                                    <div id="object-specific">
                                        {/* <!-- DON'T REMOVE --> */}
                                        <input id="temp-color" type="hidden" className="form-field colorpicker" autoComplete="off" defaultValue="#fff" />
                                    </div>
                                </div>
                            </div>
                        </div>
                        {/* <!-- PROJECTS TOOL --> */}
                        <div id="project-tool" className="d-none">
                            <div className="tool-content">
                                <p className="property-title">Templates</p>
                                <span className="collapse material-icons">
                                    keyboard_double_arrow_left
                                </span>
                                <div className="nav-tab-wrap">
                                    <div className="nav-tabs">
                                        <div className="nav-tab nav-tab-active" data-target="project-library">
                                            Template Library
                                        </div>
                                        <div className="nav-tab" data-target="my-templates">
                                            My Templates
                                        </div>
                                    </div>
                                    <div id="project-library" className="nav-tab-content">
                                        <div id="template-library" className="css-grid">
                                            <div className="image-grid-item" data-template="files/templates/1.json" title="Template 1">
                                                <div className="img-wrap">
                                                    <div className="img-loader"></div>
                                                    <img className="lazy" src="files/templates/1.png" alt="" draggable="false" />
                                                </div>
                                            </div>
                                            <div className="image-grid-item" data-template="files/templates/2.json" title="Template 2">
                                                <div className="img-wrap">
                                                    <div className="img-loader"></div>
                                                    <img className="lazy" src="files/templates/2.png" alt="" draggable="false" />
                                                </div>
                                            </div>
                                            <div className="image-grid-item" data-template="files/templates/3.json" title="Template 3">
                                                <div className="img-wrap">
                                                    <div className="img-loader"></div>
                                                    <img className="lazy" src="files/templates/3.png" alt="" draggable="false" />
                                                </div>
                                            </div>
                                            <div className="image-grid-item" data-template="files/templates/4.json" title="Template 4">
                                                <div className="img-wrap">
                                                    <div className="img-loader"></div>
                                                    <img className="lazy" src="files/templates/4.png" alt="" draggable="false" />
                                                </div>
                                            </div>
                                            <div className="image-grid-item" data-template="files/templates/5.json" title="Template 5">
                                                <div className="img-wrap">
                                                    <div className="img-loader"></div>
                                                    <img className="lazy" src="files/templates/5.png" alt="" draggable="false" />
                                                </div>
                                            </div>
                                            <div className="image-grid-item" data-template="files/templates/6.json" title="Template 6">
                                                <div className="img-wrap">
                                                    <div className="img-loader"></div>
                                                    <img className="lazy" src="files/templates/6.png" alt="" draggable="false" />
                                                </div>
                                            </div>
                                            <div className="image-grid-item" data-template="files/templates/7.json" title="Template 7">
                                                <div className="img-wrap">
                                                    <div className="img-loader"></div>
                                                    <img className="lazy" src="files/templates/7.png" alt="" draggable="false" />
                                                </div>
                                            </div>
                                            <div className="image-grid-item" data-template="files/templates/8.json" title="Template 8">
                                                <div className="img-wrap">
                                                    <div className="img-loader"></div>
                                                    <img className="lazy" src="files/templates/8.png" alt="" draggable="false" />
                                                </div>
                                            </div>
                                            <div className="image-grid-item" data-template="files/templates/9.json" title="Template 9">
                                                <div className="img-wrap">
                                                    <div className="img-loader"></div>
                                                    <img className="lazy" src="files/templates/9.png" alt="" draggable="false" />
                                                </div>
                                            </div>
                                            <div className="image-grid-item" data-template="files/templates/10.json" title="Template 10">
                                                <div className="img-wrap">
                                                    <div className="img-loader"></div>
                                                    <img className="lazy" src="files/templates/10.png" alt="" draggable="false" />
                                                </div>
                                            </div>
                                            <div className="image-grid-item" data-template="files/templates/11.json" title="Template 11">
                                                <div className="img-wrap">
                                                    <div className="img-loader"></div>
                                                    <img className="lazy" src="files/templates/11.png" alt="" draggable="false" />
                                                </div>
                                            </div>
                                            <div className="image-grid-item" data-template="files/templates/12.json" title="Template 12">
                                                <div className="img-wrap">
                                                    <div className="img-loader"></div>
                                                    <img className="lazy" src="files/templates/12.png" alt="" draggable="false" />
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div id="my-templates" className="nav-tab-content d-none">
                                        <div className="project-menu-btns">
                                            <button type="button" id="import-project" className="btn">Import</button>
                                            <button type="button" id="export-project" className="btn">Export</button>
                                            <button type="button" id="save-project" className="btn primary">Save</button>
                                        </div>
                                        <div id="projects-output"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        {/* <!-- UPLOAD TOOL --> */}
                        <div id="upload-tool" className="d-none">
                            <div className="tool-content">
                                <p className="property-title">Uploads</p>
                                <span className="collapse material-icons">
                                    keyboard_double_arrow_left
                                </span>
                                <div className="assets-buttons">
                                    <button id="upload-button" type="button" className="btn primary">
                                        <span className="material-icons">file_upload</span>
                                        Upload media
                                    </button>
                                    <button id="delete-assets-button" type="button" className="btn danger">
                                        <span className="material-icons">delete_forever</span>
                                    </button>
                                </div>
                                <div id="upload-tabs">
                                    <div id="images-tab" className="upload-tab upload-tab-active">
                                        Images
                                    </div>
                                    <div id="videos-tab" className="upload-tab">
                                        Videos
                                    </div>
                                </div>
                                <div id="uploaded-images">
                                    <div id="uploaded-images-grid" className="css-grid"></div>
                                </div>
                                <div id="uploaded-videos" className="d-none">
                                    <div id="uploaded-videos-grid" className="css-grid"></div>
                                </div>
                            </div>
                        </div>
                        {/* <!-- TEXT TOOL --> */}
                        <div id="text-tool" className="d-none">
                            <div className="tool-content">
                                <p className="property-title">Text</p>
                                <span className="collapse material-icons">
                                    keyboard_double_arrow_left
                                </span>
                                <div className="shapes-cont">
                                    <p className="row-title">Basic text</p>
                                    <div id="heading-text" data-font="Roboto" className="add-text noselect">
                                        Add a heading
                                    </div>
                                    <div id="subheading-text" data-font="Roboto" className="add-text noselect">
                                        Add a subheading
                                    </div>
                                    <div id="body-text" data-font="Roboto" className="add-text noselect">
                                        Add body text
                                    </div>
                                    <p className="row-title">Animated</p>
                                    <div className="animated-text-grid">
                                        <div className="animated-text-item noselect" data-id="fade in">
                                            <img draggable="false" className="noselect" src="assets/fade-in.svg" />
                                            <div>Fade In</div>
                                        </div>
                                        <div className="animated-text-item noselect" data-id="typewriter">
                                            <img draggable="false" className="noselect" src="assets/typewriter.svg" />
                                            <div>Typewriter</div>
                                        </div>
                                        <div className="animated-text-item noselect" data-id="slide top">
                                            <img draggable="false" className="noselect" src="assets/slide-top.svg" />
                                            <div>Slide Top</div>
                                        </div>
                                        <div className="animated-text-item noselect" data-id="slide bottom">
                                            <img draggable="false" className="noselect" src="assets/slide-bottom.svg" />
                                            <div>Slide Bottom</div>
                                        </div>
                                        <div className="animated-text-item noselect" data-id="slide left">
                                            <img draggable="false" className="noselect" src="assets/slide-left.svg" />
                                            <div>Slide Left</div>
                                        </div>
                                        <div className="animated-text-item noselect" data-id="slide right">
                                            <img draggable="false" className="noselect" src="assets/slide-right.svg" />
                                            <div>Slide Right</div>
                                        </div>
                                        <div className="animated-text-item noselect" data-id="scale">
                                            <img draggable="false" className="noselect" src="assets/scale.svg" />
                                            <div>Scale</div>
                                        </div>
                                        <div className="animated-text-item noselect" data-id="shrink">
                                            <img draggable="false" className="noselect" src="assets/shrink.svg" />
                                            <div>Shrink</div>
                                        </div>
                                    </div>
                                    <p className="row-title">Sans Serif</p>
                                    <div className="add-text noselect" data-font="Roboto" >
                                        Roboto
                                    </div>
                                    <div className="add-text noselect" data-font="Montserrat">
                                        Montserrat
                                    </div>
                                    <div className="add-text noselect" data-font="Poppins" >
                                        Poppins
                                    </div>
                                    <p className="row-title">Serif</p>
                                    <div className="add-text noselect" data-font="Playfair Display">
                                        Playfair Display
                                    </div>
                                    <div className="add-text noselect" data-font="Merriweather" >
                                        Merriweather
                                    </div>
                                    <div className="add-text noselect" data-font="IBM Plex Serif" >
                                        IBM Plex Serif
                                    </div>
                                    <p className="row-title">Monospace</p>
                                    <div className="add-text noselect" data-font="Roboto Mono" >
                                        Roboto Mono
                                    </div>
                                    <div className="add-text noselect" data-font="Inconsolata" >
                                        Inconsolata
                                    </div>
                                    <div className="add-text noselect" data-font="Source Code Pro">
                                        Source Code Pro
                                    </div>
                                    <p className="row-title">Handwriting</p>
                                    <div className="add-text noselect" data-font="Dancing Script" >
                                        Dancing Script
                                    </div>
                                    <div className="add-text noselect" data-font="Pacifico" >
                                        Pacifico
                                    </div>
                                    <div className="add-text noselect" data-font="Indie Flower" >
                                        Indie Flower
                                    </div>
                                    <p className="row-title">Display</p>
                                    <div className="add-text noselect" data-font="Lobster" >
                                        Lobster
                                    </div>
                                    <div className="add-text noselect" data-font="Bebas Neue" >
                                        Bebas Neue
                                    </div>
                                    <div className="add-text noselect" data-font="Titan One" >
                                        Titan One
                                    </div>
                                </div>
                            </div>
                        </div>
                        {/* <!-- OBJECTS TOOL --> */}
                        <div id="shape-tool" className="d-none">
                            <div className="tool-content">
                                <p className="property-title">Objects</p>
                                <span className="collapse material-icons">
                                    keyboard_double_arrow_left
                                </span>
                                <div className="shapes-cont">
                                    {/* <!-- Shapes --> */}
                                    <p className="row-title">Shapes</p>
                                    <div id="shapes-gallery-row" className="gallery-row">
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/shapes/1.svg" data-file="files/shapes/1.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/shapes/2.svg" data-file="files/shapes/2.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/shapes/3.svg" data-file="files/shapes/3.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/shapes/4.svg" data-file="files/shapes/4.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/shapes/5.svg" data-file="files/shapes/5.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/shapes/6.svg" data-file="files/shapes/6.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/shapes/7.svg" data-file="files/shapes/7.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/shapes/8.svg" data-file="files/shapes/8.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/shapes/9.svg" data-file="files/shapes/9.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/shapes/10.svg" data-file="files/shapes/10.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/shapes/11.svg" data-file="files/shapes/11.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/shapes/12.svg" data-file="files/shapes/12.svg" />
                                        </div>
                                    </div>
                                    <button type="button" className="btn btn-full object-more" data-folder="files/shapes/" data-offset="12" data-count="140">Load More</button>
                                    {/* <!-- Emojis --> */}
                                    <p className="row-title">Emojis</p>
                                    <div className="gallery-row">
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/emojis/1.svg" data-file="files/emojis/1.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/emojis/2.svg" data-file="files/emojis/2.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/emojis/3.svg" data-file="files/emojis/3.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/emojis/4.svg" data-file="files/emojis/4.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/emojis/5.svg" data-file="files/emojis/5.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/emojis/6.svg" data-file="files/emojis/6.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/emojis/7.svg" data-file="files/emojis/7.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/emojis/8.svg" data-file="files/emojis/8.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/emojis/9.svg" data-file="files/emojis/9.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/emojis/10.svg" data-file="files/emojis/10.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/emojis/11.svg" data-file="files/emojis/11.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/emojis/12.svg" data-file="files/emojis/12.svg" />
                                        </div>
                                    </div>
                                    <button type="button" className="btn btn-full object-more" data-folder="files/emojis/" data-offset="12" data-count="152">Load More</button>
                                    {/* <!-- Stickers --> */}
                                    <p className="row-title">Others</p>
                                    <div className="gallery-row">
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/others/1.svg" data-file="files/others/1.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/others/2.svg" data-file="files/others/2.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/others/3.svg" data-file="files/others/3.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/others/4.svg" data-file="files/others/4.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/others/5.svg" data-file="files/others/5.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/others/6.svg" data-file="files/others/6.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/others/7.svg" data-file="files/others/7.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/others/8.svg" data-file="files/others/8.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/others/9.svg" data-file="files/others/9.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/others/10.svg" data-file="files/others/10.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/others/11.svg" data-file="files/others/11.svg" />
                                        </div>
                                        <div className="grid-item">
                                            <img className="object-svg" draggable="false" src="files/others/12.svg" data-file="files/others/12.svg" />
                                        </div>
                                    </div>
                                    <button type="button" className="btn btn-full object-more" data-folder="files/others/" data-offset="12" data-count="48">Load More</button>
                                </div>
                            </div>
                        </div>
                        {/* <!-- IMAGE TOOL --> */}
                        <div id="image-tool" className="d-none">
                            <div className="tool-content">
                                <p className="property-title">Images</p>
                                <span className="collapse material-icons">
                                    keyboard_double_arrow_left
                                </span>
                                <div className="nav-tabs-wrap">
                                    <div className="nav-tabs">
                                        <div className="nav-tab nav-tab-active" data-target="pixabay-images">
                                            Pixabay
                                        </div>
                                        <div className="nav-tab" data-target="pexels-images">
                                            Pexels
                                        </div>
                                    </div>
                                    <div id="pixabay-images" className="nav-tab-content">
                                        <div className="browser-wrap">
                                            <div className="browser-search-box">
                                                <input id="pixabay-img-search" className="form-field" autoComplete="off" placeholder="Enter a keyword..." />
                                                <span className="material-icons delete-search">clear</span>
                                                <button id="pixabay-img-search-btn" type="button" className="btn primary">
                                                    <span className="material-icons">search</span>
                                                </button>
                                            </div>
                                            <div id="pixabay-search-options" className="d-none">
                                                <select id="pixabay-orientation" className="custom-select" autoComplete="off">
                                                    <option defaultValue="" disabled>All Orientations</option>
                                                    <option defaultValue="horizontal">Horizontal</option>
                                                    <option defaultValue="vertical">Vertical</option>
                                                </select>
                                                <select id="pixabay-color" className="custom-select" autoComplete="off">
                                                    <option defaultValue="" disabled>All Colors</option>
                                                    <option defaultValue="white">White</option>
                                                    <option defaultValue="black">Black</option>
                                                    <option defaultValue="gray">Gray</option>
                                                    <option defaultValue="grayscale">Grayscale</option>
                                                    <option defaultValue="brown">Brown</option>
                                                    <option defaultValue="blue">Blue</option>
                                                    <option defaultValue="turquoise">Turquoise</option>
                                                    <option defaultValue="red">Red</option>
                                                    <option defaultValue="lilac">Lilac</option>
                                                    <option defaultValue="pink">Pink</option>
                                                    <option defaultValue="orange">Orange</option>
                                                    <option defaultValue="yellow">Yellow</option>
                                                    <option defaultValue="green">Green</option>
                                                </select>
                                                <select id="pixabay-category" className="custom-select" autoComplete="off">
                                                    <option defaultValue="" disabled>All Categories</option>
                                                    <option defaultValue="backgrounds">Backgrounds</option>
                                                    <option defaultValue="fashion">Fashion</option>
                                                    <option defaultValue="nature">Nature</option>
                                                    <option defaultValue="science">Science</option>
                                                    <option defaultValue="education">Education</option>
                                                    <option defaultValue="feelings">Feelings</option>
                                                    <option defaultValue="health">Health</option>
                                                    <option defaultValue="people">People</option>
                                                    <option defaultValue="religion">Religion</option>
                                                    <option defaultValue="places">Places</option>
                                                    <option defaultValue="animals">Animals</option>
                                                    <option defaultValue="industry">Industry</option>
                                                    <option defaultValue="computer">Computer</option>
                                                    <option defaultValue="food">Food</option>
                                                    <option defaultValue="sports">Sports</option>
                                                    <option defaultValue="transportation">Transportation</option>
                                                    <option defaultValue="travel">Travel</option>
                                                    <option defaultValue="buildings">Buildings</option>
                                                    <option defaultValue="business">Business</option>
                                                    <option defaultValue="music">Music</option>
                                                </select>
                                            </div>
                                            <div className="stock-settings-toggle">
                                                <div id="pixabay-settings-toggle">More Settings <span className="material-icons">expand_more</span></div>
                                            </div>
                                            <div id="pixabay-img-output"></div>
                                            <div className="notice">Photos provided by <a href="https://pixabay.com/" target="_blank">Pixabay</a>.</div>
                                        </div>
                                    </div>
                                    <div id="pexels-images" className="nav-tab-content d-none">
                                        <div className="browser-wrap">
                                            <div className="browser-search-box">
                                                <input id="pexels-img-search" className="form-field" autoComplete="off" placeholder="Enter a keyword..." />
                                                <span className="material-icons delete-search">clear</span>
                                                <button id="pexels-img-search-btn" type="button" className="btn primary">
                                                    <span className="material-icons">search</span>
                                                </button>
                                            </div>
                                            <div id="pexels-search-options" className="d-none">
                                                <select id="pexels-orientation" className="custom-select" autoComplete="off" disabled>
                                                    <option defaultValue="" disabled>All Orientations</option>
                                                    <option defaultValue="landscape">Landscape</option>
                                                    <option defaultValue="portrait">Portrait</option>
                                                    <option defaultValue="square">Square</option>
                                                </select>
                                                <select id="pexels-color" className="custom-select" autoComplete="off" disabled>
                                                    <option defaultValue="" disabled>All Colors</option>
                                                    <option defaultValue="white">White</option>
                                                    <option defaultValue="black">Black</option>
                                                    <option defaultValue="gray">Gray</option>
                                                    <option defaultValue="brown">Brown</option>
                                                    <option defaultValue="blue">Blue</option>
                                                    <option defaultValue="turquoise">Turquoise</option>
                                                    <option defaultValue="red">Red</option>
                                                    <option defaultValue="violet">Violet</option>
                                                    <option defaultValue="pink">Pink</option>
                                                    <option defaultValue="orange">Orange</option>
                                                    <option defaultValue="yellow">Yellow</option>
                                                    <option defaultValue="green">Green</option>
                                                </select>
                                            </div>
                                            <div className="stock-settings-toggle">
                                                <div id="pexels-settings-toggle">More Settings <span className="material-icons">expand_more</span></div>
                                            </div>
                                            <div id="pexels-img-output"></div>
                                            <div className="notice">Photos provided by <a href="https://pexels.com/" target="_blank">Pexels</a>.</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        {/* <!-- VIDEO TOOL --> */}
                        <div id="video-tool" className="d-none">
                            <div className="tool-content">
                                <p className="property-title">Videos</p>
                                <span className="collapse material-icons">
                                    keyboard_double_arrow_left
                                </span>
                                <div className="nav-tabs-wrap">
                                    <div className="nav-tabs">
                                        <div className="nav-tab nav-tab-active" data-target="pixabay-videos">
                                            Pixabay
                                        </div>
                                        <div className="nav-tab" data-target="pexels-videos">
                                            Pexels
                                        </div>
                                    </div>
                                    <div id="pixabay-videos" className="nav-tab-content">
                                        <div className="browser-wrap">
                                            <div className="browser-search-box">
                                                <input id="pixabay-video-search" className="form-field" autoComplete="off" placeholder="Enter a keyword..." />
                                                <span className="material-icons delete-search">clear</span>
                                                <button id="pixabay-video-search-btn" type="button" className="btn primary">
                                                    <span className="material-icons">search</span>
                                                </button>
                                            </div>
                                            <div id="pixabay-video-options" className="d-none">
                                                <select id="pixabay-orientation-video" className="custom-select" autoComplete="off">
                                                    <option defaultValue="" disabled>All Orientations</option>
                                                    <option defaultValue="horizontal">Horizontal</option>
                                                    <option defaultValue="vertical">Vertical</option>
                                                </select>
                                                <select id="pixabay-color-video" className="custom-select" autoComplete="off">
                                                    <option defaultValue="" disabled>All Colors</option>
                                                    <option defaultValue="white">White</option>
                                                    <option defaultValue="black">Black</option>
                                                    <option defaultValue="gray">Gray</option>
                                                    <option defaultValue="grayscale">Grayscale</option>
                                                    <option defaultValue="brown">Brown</option>
                                                    <option defaultValue="blue">Blue</option>
                                                    <option defaultValue="turquoise">Turquoise</option>
                                                    <option defaultValue="red">Red</option>
                                                    <option defaultValue="lilac">Lilac</option>
                                                    <option defaultValue="pink">Pink</option>
                                                    <option defaultValue="orange">Orange</option>
                                                    <option defaultValue="yellow">Yellow</option>
                                                    <option defaultValue="green">Green</option>
                                                </select>
                                                <select id="pixabay-category-video" className="custom-select" autoComplete="off">
                                                    <option defaultValue="" disabled>All Categories</option>
                                                    <option defaultValue="backgrounds">Backgrounds</option>
                                                    <option defaultValue="fashion">Fashion</option>
                                                    <option defaultValue="nature">Nature</option>
                                                    <option defaultValue="science">Science</option>
                                                    <option defaultValue="education">Education</option>
                                                    <option defaultValue="feelings">Feelings</option>
                                                    <option defaultValue="health">Health</option>
                                                    <option defaultValue="people">People</option>
                                                    <option defaultValue="religion">Religion</option>
                                                    <option defaultValue="places">Places</option>
                                                    <option defaultValue="animals">Animals</option>
                                                    <option defaultValue="industry">Industry</option>
                                                    <option defaultValue="computer">Computer</option>
                                                    <option defaultValue="food">Food</option>
                                                    <option defaultValue="sports">Sports</option>
                                                    <option defaultValue="transportation">Transportation</option>
                                                    <option defaultValue="travel">Travel</option>
                                                    <option defaultValue="buildings">Buildings</option>
                                                    <option defaultValue="business">Business</option>
                                                    <option defaultValue="music">Music</option>
                                                </select>
                                            </div>
                                            <div className="stock-settings-toggle">
                                                <div id="pixabay-videos-toggle">More Settings <span className="material-icons">expand_more</span></div>
                                            </div>
                                            <div id="pixabay-video-output"></div>
                                            <div className="notice">Videos provided by <a href="https://pixabay.com/" target="_blank">Pixabay</a>.</div>
                                        </div>
                                    </div>
                                    <div id="pexels-videos" className="nav-tab-content d-none">
                                        <div className="browser-wrap">
                                            <div className="browser-search-box">
                                                <input id="pexels-video-search" className="form-field" autoComplete="off" placeholder="Enter a keyword..." />
                                                <span className="material-icons delete-search">clear</span>
                                                <button id="pexels-video-search-btn" type="button" className="btn primary">
                                                    <span className="material-icons">search</span>
                                                </button>
                                            </div>
                                            <div id="pexels-video-options" className="d-none">
                                                <select id="pexels-video-orientation" className="custom-select" autoComplete="off" disabled>
                                                    <option defaultValue="" disabled>All Orientations</option>
                                                    <option defaultValue="landscape">Landscape</option>
                                                    <option defaultValue="portrait">Portrait</option>
                                                    <option defaultValue="square">Square</option>
                                                </select>
                                            </div>
                                            <div className="stock-settings-toggle">
                                                <div id="pexels-videos-toggle">More Settings <span className="material-icons">expand_more</span></div>
                                            </div>
                                            <div id="pexels-video-output"></div>
                                            <div className="notice">Videos provided by <a href="https://pexels.com/" target="_blank">Pexels</a>.</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        {/* <!-- AUDIO TOOL --> */}
                        <div id="audio-tool" className="d-none">
                            <div className="tool-content audio-browser">
                                <p className="property-title">Audio</p>
                                <button id="audio-upload-button" type="button" className="btn btn-full primary">
                                    <span className="material-icons">file_upload</span>
                                    Upload audio
                                </button>
                                <span className="collapse material-icons">
                                    keyboard_double_arrow_left
                                </span>
                                <div id="audio-list-parent">
                                    <div id="audio-list">
                                        <div className="audio-item" data-file="files/audio/stomping-rock.mp3">
                                            <div className="audio-preview">
                                                <span className="material-icons">play_arrow</span>
                                            </div>
                                            <img className="audio-thumb" src="files/audio/stomping-rock-thumb.png" />
                                            <div className="audio-info">
                                                <div className="audio-info-title">Stomping Rock (Four Shots)</div>
                                                <div className="audio-info-duration">1:59</div>
                                            </div>
                                        </div>
                                        <div className="audio-item" data-file="files/audio/everything-feels-new.mp3">
                                            <div className="audio-preview">
                                                <span className="material-icons">play_arrow</span>
                                            </div>
                                            <img className="audio-thumb" src="files/audio/everything-feels-new-thumb.png" />
                                            <div className="audio-info">
                                                <div className="audio-info-title">Everything Feels New</div>
                                                <div className="audio-info-duration">1:06</div>
                                            </div>
                                        </div>
                                        <div className="audio-item" data-file="files/audio/the-podcast-intro.mp3">
                                            <div className="audio-preview">
                                                <span className="material-icons">play_arrow</span>
                                            </div>
                                            <img className="audio-thumb" src="files/audio/the-podcast-intro-thumb.png" />
                                            <div className="audio-info">
                                                <div className="audio-info-title">The Podcast Intro</div>
                                                <div className="audio-info-duration">1:51</div>
                                            </div>
                                        </div>
                                        <div className="audio-item" data-file="files/audio/epic-cinematic-trailer.mp3">
                                            <div className="audio-preview">
                                                <span className="material-icons">play_arrow</span>
                                            </div>
                                            <img className="audio-thumb" src="files/audio/epic-cinematic-trailer-thumb.png" />
                                            <div className="audio-info">
                                                <div className="audio-info-title">Epic Cinematic Trailer</div>
                                                <div className="audio-info-duration">2:27</div>
                                            </div>
                                        </div>
                                        <div className="audio-item" data-file="files/audio/inspirational-background.mp3">
                                            <div className="audio-preview">
                                                <span className="material-icons">play_arrow</span>
                                            </div>
                                            <img className="audio-thumb" src="files/audio/inspirational-background-thumb.png" />
                                            <div className="audio-info">
                                                <div className="audio-info-title">Inspirational Background</div>
                                                <div className="audio-info-duration">2:19</div>
                                            </div>
                                        </div>
                                        <div className="audio-item" data-file="files/audio/tropical-summer-music.mp3">
                                            <div className="audio-preview">
                                                <span className="material-icons">play_arrow</span>
                                            </div>
                                            <img className="audio-thumb" src="files/audio/tropical-summer-music-thumb.png" />
                                            <div className="audio-info">
                                                <div className="audio-info-title">Tropical Summer Music</div>
                                                <div className="audio-info-duration">2:35</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div className="landing-text">
                                    Browse millions of assets from Pixabay by
                                    <a href="https://pixabay.com/music/" target="_blank">
                                        clicking here.
                                    </a>
                                </div>
                            </div>
                        </div>
                        {/* <!-- QR CODE TOOL --> */}
                        <div id="qrcode-tool" className="d-none">
                            <div className="tool-content">
                                <p className="property-title">QR CODE</p>
                                <span className="collapse material-icons">
                                    keyboard_double_arrow_left
                                </span>
                                <div id="qrcode-settings">
                                    <div className="control-wrap">
                                        <label className="control-label">Preview</label>
                                        <div className="control">
                                            <div id="qrcode-preview"></div>
                                        </div>
                                    </div>
                                    <hr />
                                    <div className="control-wrap">
                                        <label className="control-label">Text</label>
                                        <div className="control">
                                            <input type="text" id="qrcode-text" className="form-field"
                                                autoComplete="off" defaultValue="https://mysite.com" />
                                        </div>
                                    </div>
                                    <div className="control-wrap control-text-color">
                                        <label className="control-label">Fill Color</label>
                                        <div className="control">
                                            <div className="colorpicker-box"></div>
                                            <input id="qrcode-fill" className="form-field colorpicker app-colorpicker" autoComplete="off" defaultValue="#333333" readOnly />
                                        </div>
                                    </div>
                                    <div className="control-wrap control-text-color">
                                        <label className="control-label">Background</label>
                                        <div className="control">
                                            <div className="colorpicker-box"></div>
                                            <input id="qrcode-back" className="form-field colorpicker app-colorpicker" autoComplete="off" defaultValue="#FFFFFF" readOnly />
                                        </div>
                                    </div>
                                    <div className="control-wrap">
                                        <label className="control-label">Rounded Corners</label>
                                        <div className="control">
                                            <input id="qrcode-rounded" type="range" min="0" max="100" defaultValue="0" step="1"
                                                className="rangeslider" autoComplete="off" />
                                        </div>
                                    </div>
                                    <hr />
                                    <div className="control-wrap">
                                        <label className="control-label">Label</label>
                                        <div className="control">
                                            <input type="text" id="qrcode-label" className="form-field"
                                                autoComplete="off" defaultValue="" />
                                        </div>
                                    </div>
                                    <div className="control-wrap control-text-color">
                                        <label className="control-label">Label Color</label>
                                        <div className="control">
                                            <div className="colorpicker-box"></div>
                                            <input id="qrcode-label-color" className="form-field colorpicker app-colorpicker" autoComplete="off" defaultValue="#333333" readOnly />
                                        </div>
                                    </div>
                                    <div className="control-wrap">
                                        <label className="control-label slider-label">Label Size</label>
                                        <div className="control">
                                            <input id="qrcode-label-size" type="range" min="0" max="100" defaultValue="30" step="1"
                                                className="rangeslider" autoComplete="off" />
                                        </div>
                                    </div>
                                    <div className="control-wrap">
                                        <label className="control-label slider-label">Position X</label>
                                        <div className="control">
                                            <input id="qrcode-label-position-x" type="range" min="0" max="100" defaultValue="50"
                                                step="1" className="rangeslider" autoComplete="off" />
                                        </div>
                                    </div>
                                    <div className="control-wrap">
                                        <label className="control-label slider-label">Position Y</label>
                                        <div className="control">
                                            <input id="qrcode-label-position-y" type="range" min="0" max="100" defaultValue="50"
                                                step="1" className="rangeslider" autoComplete="off" />
                                        </div>
                                    </div>
                                </div>
                                <hr />
                                <button id="generate-qr-code" type="button" className="btn primary lg-btn btn-full">Generate QR Code</button>
                            </div>
                        </div>
                    </div>
                </div>
                {/* <!-- BROWSER END -->
                <!-- CANVAS AREA --> */}
                <div id="canvas-area">
                    {/* <!-- TOP CANVAS CONTROLS --> */}
                    <div id="top-canvas-controls">
                        <div id="clear-project">
                            <span className="material-icons">delete</span>
                            <span className="sm-hide">Clear</span>
                        </div>
                        <div id="undo">
                            <span className="material-icons">undo</span>
                            <span className="sm-hide">Undo</span>
                        </div>
                        <div id="redo">
                            <span className="material-icons">redo</span>
                            <span className="sm-hide">Redo</span>
                        </div>
                    </div>
                    {/* <!-- TOP CANVAS MENU --> */}
                    {/* <div id="top-canvas-menu">
                        <div id="theme-switcher" className="dark">
                            <span className="material-icons">wb_sunny</span>
                        </div>
                        <div id="user-menu" className="user-menu">
                            <div className="dropdown-wrap">
                                <img alt="avatar" src='assets/avatar.png' /><span className="material-icons">arrow_drop_down</span>
                                <div className="menu-menu-container">
                                    <ul className="dropdown">
                                        <li><a href="#"><span className="material-icons">home</span>Home</a></li>
                                        <li><a href="#"><span className="material-icons">article</span>Documentation</a>
                                        </li>
                                        <li><a href="#"><span className="material-icons">email</span>Contact Us</a></li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div> */}
                    {/* <!-- BOTTOM CANVAS --> */}
                    <div id="bottom-canvas">
                        <div id="zoom-options">
                            <div title="Hand tool" id="hand-tool">
                                <span className="material-icons">pan_tool</span>
                            </div>
                            <div id="zoom-level">
                                <div className="zoom-item" data-zoom="out" title="Zoom out">
                                    <span className="material-icons">remove</span>
                                </div>
                                <div className="zoom-span">100%</div>
                                <div className="zoom-item" data-zoom="in" title="Zoom in">
                                    <span className="material-icons">add</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    {/* <!-- IMAGE LOADER --> */}
                    <div id="load-image" className="load-media">
                        <div className="load-media-wrap">
                            <div className="canvas-loader"></div>
                            <span>Loading image...</span>
                        </div>
                    </div>
                    {/* <!-- VIDEO LOADER --> */}
                    <div id="load-video" className="load-media">
                        <div className="load-media-wrap">
                            <div className="canvas-loader"></div>
                            <span>Loading video...</span>
                        </div>
                    </div>
                    {/* <!-- TEMPLATE LOADER --> */}
                    <div id="load-template" className="load-media">
                        <div className="load-media-wrap">
                            <div className="canvas-loader"></div>
                            <span>Loading template...</span>
                        </div>
                    </div>
                    {/* <!-- CANVAS --> */}
                    <canvas id="canvas"></canvas>
                    {/* <!-- Canvas for recording --> */}
                    <div className="d-none">
                        <canvas id="canvasrecord"></canvas>
                    </div>
                </div>
                {/* <!-- CANVAS AREA END --> */}
                {/* <!-- Timeline Handle --> */}
                <div id="timeline-handle"></div>
                {/* <!-- BOTTOM AREA --> */}
                <div id="bottom-area" className="noselect">
                    <div id="keyframe-properties">
                        <div id="easing">
                            <p className="property-title">Keyframe easing</p>
                            <select className="custom-select">
                                <option defaultValue="linear">Linear</option>
                                <option defaultValue="easeInQuad">Ease in</option>
                                <option defaultValue="easeOutQuad">Ease out</option>
                                <option defaultValue="easeInOutQuad">Ease in-out</option>
                                <option defaultValue="easeOutInQuad">Ease out-in</option>
                                <option defaultValue="easeInBounce">Ease in bounce</option>
                                <option defaultValue="easeOutBounce">Ease out bounce</option>
                                <option defaultValue="easeInOutBounce">Ease in-out bounce</option>
                                <option defaultValue="easeOutInBounce">Ease out-in bounce</option>
                                <option defaultValue="easeInSine">Ease in sine</option>
                                <option defaultValue="easeOutSine">Ease out sine</option>
                                <option defaultValue="easeInOutSine">Ease in-out sine</option>
                                <option defaultValue="easeOutInSine">Ease out-in sine</option>
                                <option defaultValue="easeInCubic">Ease in cubic</option>
                                <option defaultValue="easeOutCubic">Ease out cubic</option>
                                <option defaultValue="easeInOutCubic">Ease in-out cubic</option>
                                <option defaultValue="easeOutInCubic">Ease out-in cubic</option>
                            </select>
                            <button id="delete-keyframe" type="button" className="btn btn-full danger">Delete Keyframe</button>
                        </div>
                    </div>
                    <div id="nothing"></div>
                    <div id="layer-list">
                        <div id="layerhead"><span className="material-icons">layers</span>LAYERS</div>
                        <div id="layer-inner-list" className="nolayers">
                            <div id="nolayers">
                                <h6>No Layers</h6>
                                <p>Add an object to get started...</p>
                            </div>
                        </div>
                    </div>
                    <div id="timearea">
                        <div id="timeline">
                            <div id="seekarea">
                                <div id="inner-seekarea">
                                    <div id="seekevents"></div>
                                </div>
                                <div id="time-numbers" className="noselect"></div>
                                <div id="seek-hover"></div>
                                <div id="seekbar"></div>
                            </div>
                            <div id="line-snap"></div>
                            <div id="inner-timeline"></div>
                        </div>
                    </div>
                </div>
                {/* <!-- BOTTOM AREA END -->
                <!-- TIMELINE CONTROLS --> */}
                <div id="controls" className="noselect">
                    {/* <div id="timeline-wrap">
                        <span className="material-icons timeline-icon">view_timeline</span>
                        <input id="timeline-zoom" type="range" min="5" max="47" defaultValue="47" step="1" className="rangeslider" autoComplete="off" />
                        <div id="speed">
                            <div id="speed-settings">
                                <div className="speed" data-speed="4">
                                    4.0x
                                </div>
                                <div className="speed" data-speed="3">
                                    3.0x
                                </div>
                                <div className="speed" data-speed="2">
                                    2.0x
                                </div>
                                <div className="speed" data-speed="1.5">
                                    1.5x
                                </div>
                                <div className="speed" data-speed="1">
                                    1.0x
                                </div>
                                <div className="speed" data-speed="0.5">
                                    0.5x
                                </div>
                            </div>
                            <span className="material-icons">bolt</span>
                            <span id="speed-text">1.0x</span>
                            <span id="speed-arrow" className="material-icons">expand_more</span>
                        </div>
                    </div>
                    <div id="playback">
                        <div id="current-time">
                            <input autoComplete="off" defaultValue="00:00:00" readOnly />
                        </div>
                        <span id="skip-backward" className="material-icons">skip_previous</span>
                        <span id="play-button" className="material-icons">play_arrow</span>
                        <span id="skip-forward" className="material-icons">skip_next</span>
                        <div id="total-time">
                            <input autoComplete="off" defaultValue="00:00:00" readOnly />
                        </div>
                    </div> */}
                    <div id="controls-right">
                        <button type="button" id="download" className="btn primary">
                            <span className="material-icons">download</span>
                            Upload Image
                        </button>
                    </div>
                </div>
                {/* <!-- TIMELINE CONTROLS END --> */}
                {/* <!-- UPLOADER --> */}
                <input autoComplete="off" id="emptyInput" defaultValue=" " className="o-none" />
                <input autoComplete="off" type="file" id="filepick" accept="image/*,video/*,audio/*" multiple />
                <input autoComplete="off" type="file" id="filepick2" accept="audio/*" />
                <input autoComplete="off" type="file" id="filepick3" accept="application/json" />
                <input autoComplete="off" type="file" id="import" className="d-none" accept=".json" aria-hidden="true" />
                <div id="upload-popup">
                    <div id="upload-popup-container">
                        <div id="upload-popup-header">
                            <div id="upload-popup-title">Upload media</div>
                            <span id="upload-popup-close" className="material-icons">close</span>
                        </div>
                        <div id="upload-drop-area">
                            <div id="upload-drop-group">
                                <span className="material-icons">file_upload</span>
                                <div id="upload-drop-title">Click to upload</div>
                                <div id="upload-drop-subtitle">Or drag and drop a file</div>
                            </div>
                        </div>
                        <div id="upload-link">
                            <input autoComplete="off" id="upload-link-input" placeholder="Paste an image or a video URL" />
                            <div id="upload-link-add">Add</div>
                        </div>
                    </div>
                    <div id="upload-overlay"></div>
                </div>
                {/* <!-- UPLOADER END -->
                                    <!-- DOWNLOAD MODAL --> */}
                <div id="download-modal" className="bottom-modal">
                    {/* <div id="download-progress">
                        <div className="download-progress">
                            <div id="download-progress-title">Rendering Video...</div>
                            <div id="download-progress-bar"><div className="progress-bar"><div id="progress-bar-width"></div></div></div>
                            <div id="download-progress-desc">5%</div>
                        </div>
                    </div>
                    <div id="download-gif-progress">
                        <div className="download-progress">
                            <div id="download-gif-progress-preview"><img id="download-gif-preview" src="" /></div>
                            <div id="download-gif-progress-desc">Please wait...</div>
                        </div>
                    </div> */}
                    <div id="download-options">
                        {/* <div>
                            <input autoComplete="off" className="magic-radio" type="radio" name="download-radio" id="webm-format" defaultValue="webm" defaultChecked />
                            <label htmlFor="webm-format" className="magic-radio-label">WebM Video <span className="download-badge bg-success">Recommended</span></label>
                        </div>
                        <div>
                            <div>
                                <input autoComplete="off" className="magic-radio" type="radio" name="download-radio" defaultValue="gif" id="gif-format" />
                                <label htmlFor="gif-format" className="magic-radio-label">Animated GIF <span className="download-badge bg-warning">Slowest</span></label>
                            </div>
                            <div id="gif-fps-select" className="download-sub-settings d-none">
                                <label className="download-options-label">Quality (1-256):</label>
                                <input id="gif-fps" className="form-field" autoComplete="off" min="1" max="256" type="number" defaultValue="10" />
                            </div>
                        </div> */}
                        <div>
                            <p className='popup'>
                                Click on the button below to confirm the submission of this image. This image
                                will be uploaded as the banner image for your project. Click below to proceed.
                            </p>
                            <div>
                                <input autoComplete="off" className="magic-radio" type="radio" name="download-radio" checked defaultValue="image" id="image-format" />
                                <label htmlFor="image-format" className="magic-radio-label">Image</label>
                            </div>
                            <div id="image-format-select" className="download-sub-settings d-none">
                                <label className="download-options-label">File Format:</label>
                                <select id="image-select" className="custom-select" autoComplete="off">
                                    <option defaultValue="png" selected >PNG</option>
                                    <option defaultValue="jpeg">JPEG</option>
                                    <option defaultValue="webp">WebP</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    <button type="button" className="btn primary btn-full" id="download-real">Confirm and Upload  </button>
                </div>
                {/* <!-- DOWNLOAD MODAL END -->
                                    <!-- Background overlay --> */}
                <div id="background-overlay"></div>

            </main>
        </>
    )
}

export default BannerCreator;