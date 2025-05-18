import React, { useState } from "react";
import WhatsAppIcon from "@mui/icons-material/WhatsApp";
import EmailIcon from "@mui/icons-material/Email";
import HtmlIcon from "@mui/icons-material/Html";
import FormatListBulletedIcon from "@mui/icons-material/FormatListBulleted";
import InfoIcon from "@mui/icons-material/Info";
import DeleteIcon from "@mui/icons-material/Delete";
import axios from 'axios';
import { useNotification } from "../../../context/NotificationContext";
import apibaseurl from '../../../apiConfig/api';

const ActionIcons = {
    whatsapp: <WhatsAppIcon style={{ color: "#25D366" }} />,
    mail: <EmailIcon style={{ color: "#1E88E5" }} />,
    customhtml: <HtmlIcon style={{ color: "#FF9800" }} />,
    optionform: <FormatListBulletedIcon style={{ color: "#9C27B0" }} />,
    product: <InfoIcon style={{ color: "#26A69A" }} />,
};

const SavedAction = ({ actions, setSelectedItem, billboardId, handleDeleteAction }) => {

    const [actionData, setactionData] = useState(actions);
    const { showNotification } = useNotification()

    //    Api Call to delete the action
    const handleDelete = async (id) => {
        try {
            const response = await axios.delete(`${apibaseurl}user/delete-action/${id}/${billboardId}`,{withCredentials : true})
            if (response.status === 200) {
                const updatedActions = actionData.filter((action) => action.actionId !== id);
                setactionData(updatedActions);
                handleDeleteAction(id)
                showNotification('Action removed successfully')
            }
        } catch (error) {
            console.log(error)
            showNotification('Sorry, Failed to remove action, Due to Server Error')
        }
        console.log(actions)

    };

    return (
        <div
            className="p-6 bg-white rounded-lg shadow-lg border border-gray-200 w-full mx-auto"
        >
            <h3 className="text-lg text-gray-700 mb-4 border-b pb-1">
                Saved Actions
            </h3>
            {actionData.map((data, index) => (
                <div
                    key={index}
                    className="flex items-center justify-between gap-4 mb-4 px-3 py-1 rounded-lg bg-gray-50 hover:bg-gray-100 transition-all duration-200 shadow-sm"
                >
                    {/* Action Number */}
                    <span className="text-vase  text-gray-600">
                        Action {index + 1}
                    </span>

                    {/* Icon and Action Type */}
                    <div className="flex justify-start items-center w-32 gap-2">
                        <span className="text-2xl">
                            {ActionIcons[data.actionType] || (
                                <span className="text-gray-400">?</span>
                            )}
                        </span>
                        <p className="text-gray-700 font-medium capitalize hover:underline" onClick={() => setSelectedItem(data)} >
                            {data.actionType}
                        </p>
                    </div>

                    {/* Delete Button */}
                    <button
                        onClick={() => handleDelete(data.actionId)}
                        className="text-red-500 hover:text-red-700 transition duration-200"
                        title="Delete"
                    >
                        <DeleteIcon />
                    </button>
                </div>
            ))}
        </div>
    );
};

export default SavedAction;
