import React, { useState, useEffect } from "react";
import apibaseurl from "../../apiConfig/api";
import Sidebar from "../../components/Sidebar";
import DeleteForeverIcon from '@mui/icons-material/DeleteForever';
import AddIcon from '@mui/icons-material/Add';
import axios from "axios";
import { useNotification } from "../../context/NotificationContext";
import CodeOffIcon from '@mui/icons-material/CodeOff';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import { useNavigate } from "react-router-dom";

const Post = () => {
    const [images, setImages] = useState([]);
    const { showNotification } = useNotification();
    const [selectedEmbedCode, setSelectedEmbedCode] = useState("");
    const [isPopupOpen, setIsPopupOpen] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        (async () => {
            try {
                const response = await axios.get(`${apibaseurl}user/getCampaigns`,{withCredentials : true});
                if (response.status === 200) {
                    // setImages(response.data);
                    setImages(response.data.campaigns)
                }
            } catch (error) {
                showNotification('Sorry, some error occurred. Please try later.');
            }
        })();
    }, []);

    const handleDelete = async (id) => {
        try {
            const response = await axios.delete(`${apibaseurl}user/delete-billboard/${id}`,{withCredentials : true});
            if (response.status === 200) {
                setImages(images.filter((image) => image.id !== id));
                showNotification("Image deleted successfully");
            }
        } catch (error) {
            console.error("Error deleting the image:", error);
            showNotification("Failed to delete the image");
        }
    };

    const handleCodeIconClick = (image) => {
        const embedCode = `<iframe width="560" height="315" src="${image.link.replace(
            "watch?v=",
            "embed/"
        )}" title="${image.title}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>`;
        setSelectedEmbedCode(embedCode);
        setIsPopupOpen(true);
    };

    const handleCopy = () => {
        navigator.clipboard.writeText(selectedEmbedCode);
        showNotification("Embed code copied to clipboard!");
    };

    const closePopup = () => {
        setIsPopupOpen(false);
        setSelectedEmbedCode("");
    };

    return (
        <>
            <Sidebar />
            <div className="min-h-screen bg-gray-100 p-6 ml-28">
                <h1 className="text-2xl font-bold mb-6 text-gray-800">Your Posts</h1>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                    {images.map((item, index) => (
                        <div
                            key={index}
                            className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-200"
                        >
                            <img
                                src={item.final_image}
                                alt={`Saved Image ${index + 1}`}
                                className="w-full h-44 object-cover"
                                onClick= {() => navigate('/campaign', {state : {billboardId : item.id}})}
                            />
                            <div className="p-4">
                                <h2 className="text-lg font-semibold text-gray-800 truncate">{item.title}</h2>
                                {/* <p className="text-sm text-gray-600 mt-2 truncate">{image.description}</p> */}
                                <div className="flex justify-between items-center mt-4">
                                    <button
                                        onClick={() => handleCodeIconClick(item)}
                                        className="px-3 py-2 bg-purple-500 text-white rounded-md hover:bg-purple-600 transition duration-200"
                                    >
                                        <CodeOffIcon fontSize="small" />
                                    </button>
                                    <button
                                        onClick={() => handleDelete(item.id)}
                                        className="px-3 py-2 bg-red-500 text-white rounded-md hover:bg-red-600 transition duration-200"
                                    >
                                        <DeleteForeverIcon fontSize="small" />
                                    </button>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                {isPopupOpen && (
                    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                        <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-md">
                            <h2 className="text-xl font-bold mb-4 text-gray-800">Embed Code</h2>
                            <div
                                className="w-full p-2 border rounded-md mb-4 bg-gray-100 text-sm font-mono whitespace-pre-wrap"
                                style={{ wordBreak: "break-word", fontFamily: 'system-ui' }}
                            >
                                {selectedEmbedCode}
                            </div>
                            <div className="flex justify-end mt-4">
                                <button
                                    onClick={handleCopy}
                                    className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition duration-200 flex items-center"
                                >
                                    <ContentCopyIcon fontSize="small" className="mr-1" /> Copy
                                </button>
                                <button
                                    onClick={closePopup}
                                    className="ml-2 px-4 py-2 bg-gray-300 text-gray-800 rounded-md hover:bg-gray-400 transition duration-200"
                                >
                                    Close
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
            
        </>
    );
};

export default Post;


