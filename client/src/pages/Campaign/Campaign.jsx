import React, { useEffect, useState } from 'react';
import apibaseurl from '../../apiConfig/api';
import { useNavigate, useLocation } from "react-router-dom";
import axios from 'axios';
import ReactQuill from 'react-quill';
import 'react-quill/dist/quill.snow.css';
import { useNotification } from '../../context/NotificationContext';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import { Info as InfoIcon, Link as LinkIcon, Description as DescriptionIcon, Html as HtmlIcon, WhatsApp as WhatsAppIcon, Mail as MailIcon, ListAlt as ListAltIcon } from '@mui/icons-material';
import { MenuItem, FormControl, InputLabel, Select, Box } from '@mui/material';
import Popup from './_component/Popup';
import AllOutIcon from '@mui/icons-material/AllOut';
import SavedAction from './_component/SavedAction';



const Campaign = () => {

  const location = useLocation();
  const billboardId = location.state?.billboardId;
  const navigate = useNavigate()
  const { showNotification } = useNotification()
  const [imageType, setimageType] = useState('png')
  const [currentStep, setcurrentStep] = useState(1)
  const [campaignInfo, setCampaignInfo] = useState(null);

  // Managing States for marking coordinate
  const [isTracking, setIsTracking] = useState(false);
  const [currentCoordinates, setCurrentCoordinates] = useState({ x: 0, y: 0 });
  const [actions, setActions] = useState([]);
  const [currentAction, setcurrentAction] = useState({
    coordinate: currentCoordinates,
    actionType: "",
    textFormat: "",
    customhtml: "",
    whatsappnumber: "",
    mail: "",
    title: "",
    link: "",
    description: "",
  });
  const [coordinatesLinks, setCoordinatesLinks] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);

  const getBillboardInfo = async () => {
    try {
      const response = await axios.post(
        `${apibaseurl}user/get-campaign`,
        {
          campaignId : billboardId
        },
        {
          withCredentials: true,
        }
      );
      if (response.status === 200) {
        setCampaignInfo(response.data.campaign);
        const existingActions = [];
        response.data.actions.map((item) => {
          existingActions.push({ coordinate: { x: item.x_coordinate, y: item.y_coordinate }, actionType: item.action_type, ...item.action_data,actionId : item.id });
        })
        console.log(existingActions);
        setActions(existingActions);
      }
    }
    catch (error) {
      console.log("No Campaign Found")
    }
  }

  useEffect(() => {
    getBillboardInfo();
  }, []);


  // const [urlInput, setUrlInput] = useState(''); 

  // Funtion to handle form input field
  const handleChange = (e) => {
    const { name, value } = e.target;
    setcurrentAction((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  // Funtion to handle change of description filed
  const handleOptionformChange = (value) => {
    setcurrentAction((prevData) => ({
      ...prevData,
      description: value,
    }));
  };

  // Funtion to handle change of image type
  const handleImageTypeChange = (e) => {
    setimageType(e.target.value)
  }

  // Funtion to save the details in the database 
  const handleSubmit = async (e) => {
    showNotification('Image details saved successfully!');
    navigate('/dashboard');
  };


  const handleSaveAction = async () => {
    if (!currentAction.actionType) {
      showNotification("Please fill out the action type before saving.");
      return;
    }
    try {
      const body = { billboardId, x: currentCoordinates.x, y: currentCoordinates.y, action_type: currentAction.actionType, action_data: currentAction }
      const response = await axios.post(`${apibaseurl}user/save-action`, body, { withCredentials: true });
      if (response.status === 201) {
        setActions((prev) => [...prev, { ...currentAction, actionId: response.data.actionId }]);
        setcurrentAction({
          coordinate: { x: 0, y: 0 },
          actionType: "",
          textFormat: "",
          customhtml: "",
          whatsappnumber: "",
          mail: "",
          title: "",
          link: "",
          description: "",
        });
        setCurrentCoordinates({ x: 0, y: 0 })
        showNotification('Action Added Sucessfully')
        setIsTracking(false);
      }
    } catch (error) {
      showNotification("Error Saving Action" + error.message);
    }
  }


  // Funtion to download Preview Image
  const handleDownload = async () => {
    if (!campaignInfo || !campaignInfo.final_image) {
      showNotification('No Image Found To Download');
      return;
    }
    try {
      const response = await fetch(campaignInfo.final_image);
      if (!response.ok) {
        showNotification('Failed to fetch image from the server.');
        return;
      }

      // Convert the response to a Blob
      const blob = await response.blob();

      // Create a URL for the Blob
      const blobUrl = URL.createObjectURL(blob);

      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = `processed-image.${imageType}`;
      link.click();
    }
    catch (error) {
      showNotification("Error while downloading image")
    }
  };


  // Function to handle the click on the image and capture the coordinates
  const handleImageClick = (e) => {
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setCurrentCoordinates({ x, y });
    setcurrentAction((prev) => ({
      ...prev,
      coordinate: { x, y },
    }));
  };


  const clickSavedAction = (item) => {
    setSelectedItem(item);
  };


  const closePopup = () => {
    setSelectedItem(null);
  };

  const handleDeleteAction = (actionId) => {
    const updatedActions = actions.filter((action) => action.actionId !== actionId);
    setActions(updatedActions);
  };

  return (
    <div className="bg-white min-h-screen flex flex-col items-center p-4">
      <div className="flex items-center flex-col w-full max-w-7xl ">
        {/* Left Section with Image */}
        {currentStep == 1 && (
          <div className="flex-1 bg-gray-100 w-1/2 p-4 rounded-lg shadow-lg">
            <div className="text-center mb-4">
              <h2 className="text-2xl ">Campaign Image</h2>
              <div className="mt-4 px-8 ">
                <img src={campaignInfo && campaignInfo.final_image} alt="Billboard Preview" className=" w-full rounded-md block mx-auto" />

              </div>
              <div className="mt-4 ">
                <button type="button" className="bg-purple-600 text-white py-2 px-7 rounded-md hover:bg-gray-400 focus:outline-none"
                  onClick={handleDownload} >
                  Download
                </button>
                <select
                  name="imageType"
                  value={imageType}
                  onChange={handleImageTypeChange}
                  className="mt-2 p-2 w-fit rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-600"
                >
                  <option value="jpg">JPG</option>
                  <option value="png">PNG</option>
                  <option value="jpeg">JPEG</option>
                </select>
                <button type="button" className="bg-purple-600 text-white py-2 px-7 rounded-md hover:bg-gray-400 ml-8 focus:outline-none"
                  onClick={() => setcurrentStep(2)}  >
                  Edit Interactivity
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Right Section with Input Fields */}
        {currentStep == 2 && (
          <>
            <h1 className='text-[27px] underline text-gray-700'>Edit image Interactivity</h1>
            <div className="flex justify-evenly gap-4 p-4 w-full bg-gray-50 rounded-lg shadow-lg">
              <div className="mt-4 w-1/2 mx-8 relative">
                {/* Image to mark coordinates */}
                <img src={campaignInfo && campaignInfo.final_image} alt="Billboard Image" className=" w-full rounded-md block mx-auto" onClick={handleImageClick} />
                {/* Marking Points where action have been added */}
                {actions.map((item, index) => (
                  <div
                    key={index}
                    className="absolute"
                    style={{
                      left: `${item.coordinate.x}px`,
                      top: `${item.coordinate.y}px`,
                    }}
                  >
                    <div className=" rounded-full absolute transform -translate-x-2 -translate-y-2">
                      <AllOutIcon style={{ fontSize: '35px', color: 'red' }} onClick={() => clickSavedAction(item)} />
                    </div>
                  </div>
                ))}

                {/* Pointer to highlight the area to add action */}
                <div
                  className="absolute"
                  style={{
                    left: `${currentCoordinates?.x}px`,
                    top: `${currentCoordinates?.y}px`,
                    visibility: `${isTracking ? 'visible' : 'hidden'}`
                  }}
                >
                  <div className=" rounded-full absolute transform -translate-x-2 -translate-y-2">
                    <Brightness7Icon style={{ fontSize: '35px', color: 'blue' }} />
                  </div>
                </div>
              </div>

              {/* Pop-Up Model to Show Action Details */}
              {selectedItem && (
                <Popup
                  data={selectedItem}
                  onClose={closePopup}
                />
              )}

              <div className="w-full max-w-lg mx-auto p-6 bg-white rounded-lg shadow-md">

                {/* Button to Add Actions */}
                <button
                  onClick={() => setIsTracking(true)}
                  className="w-full text-sm font-medium bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-6 py-2 rounded-lg shadow-md transform transition-all duration-300 hover:scale-105 hover:shadow-xl focus:outline-none focus:ring-4 focus:ring-purple-300 mb-4"
                >
                  {isTracking ? 'Note: Click on the image area to add an action' : 'Add Action'}
                </button>

                {/* Saved Action Component to view saved action */}
                {!isTracking && actions.length > 0 && (
                  <SavedAction actions={actions} setSelectedItem={setSelectedItem} billboardId={billboardId} handleDeleteAction={handleDeleteAction} />
                )}

                {/* Additional Options to Add Actions */}
                {isTracking && (
                  <div className='flex flex-col justify-between h-full'>
                    <div className="mt-6">

                      {/* Dropdown to select action type*/}
                      <div className="mb-6">
                        <FormControl fullWidth variant="outlined" className="w-[60%]">
                          <InputLabel id="section-action-label" className="text-gray-700">
                            Select The Action Type To Perform
                          </InputLabel>
                          <Select
                            labelId="section-action-label"
                            id="section-action"
                            name="actionType"
                            value={currentAction.actionType}
                            onChange={handleChange}
                            label="Select The Action Type To Perform"
                            className="bg-white shadow-sm rounded-lg   "
                          >
                            <MenuItem value="" disabled>
                              <em>Select Action</em>
                            </MenuItem>
                            <MenuItem value="whatsapp">
                              <Box display="flex" alignItems="center">
                                <WhatsAppIcon className="mr-2 text-green-600" />
                                WhatsApp
                              </Box>
                            </MenuItem>
                            <MenuItem value="mail">
                              <Box display="flex" alignItems="center">
                                <MailIcon className="mr-2 text-blue-600" />
                                Email
                              </Box>
                            </MenuItem>
                            <MenuItem value="customhtml">
                              <Box display="flex" alignItems="center">
                                <HtmlIcon className="mr-2 text-orange-600" />
                                Custom HTML
                              </Box>
                            </MenuItem>
                            <MenuItem value="optionform">
                              <Box display="flex" alignItems="center">
                                <ListAltIcon className="mr-2 text-purple-600" />
                                Option Form
                              </Box>
                            </MenuItem>
                            <MenuItem value="product">
                              <Box display="flex" alignItems="center">
                                <InfoIcon className="mr-2 text-teal-600" />
                                Product Information
                              </Box>
                            </MenuItem>
                          </Select>
                        </FormControl>
                      </div>

                      {/* Input Form to fill the details based on selected actiontype */}

                      <div className="mt-4 p-6 border border-dashed border-gray-300 rounded-lg bg-gray-50 text-gray-700 shadow-sm mb-8">

                        {/* Product actiontype details */}
                        {currentAction.actionType === 'product' && (
                          <>
                            <h3 className="text-lg  text-purple-700 mb-5 text-left flex items-center">
                              Product Details
                            </h3>
                            <div className="mb-4">
                              <label htmlFor="title" className=" text-sm font-medium text-gray-700 mb-2 flex items-center">
                                <InfoIcon className="mr-2 text-gray-500" />
                                Title
                              </label>
                              <input
                                type="text"
                                id="title"
                                name="title"
                                value={currentAction.title}
                                onChange={handleChange}
                                className="w-full p-2 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-600"
                              />
                            </div>

                            <div className="mb-4">
                              <label htmlFor="link" className=" text-sm font-medium text-gray-700 mb-2 flex items-center">
                                <LinkIcon className="mr-2 text-gray-500" />
                                Link
                              </label>
                              <input
                                type="url"
                                id="link"
                                name="link"
                                value={currentAction.link}
                                onChange={handleChange}
                                className="w-full p-2 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-600"
                              />
                            </div>

                            <div className="mb-4">
                              <label htmlFor="description" className=" text-sm font-medium text-gray-700 mb-2 flex items-center">
                                <DescriptionIcon className="mr-2 text-gray-500" />
                                Description
                              </label>
                              <ReactQuill
                                value={currentAction.description}
                                onChange={handleOptionformChange}
                                className="w-full mt-2 rounded-md border border-gray-300 focus:ring-2 focus:ring-purple-600"
                                modules={{
                                  toolbar: [
                                    [{ header: '1' }, { header: '2' }, { font: [] }],
                                    [{ list: 'ordered' }, { list: 'bullet' }],
                                    ['bold', 'italic', 'underline'],
                                    [{ align: [] }],
                                    ['blockquote'],
                                    [{ color: [] }],
                                  ],
                                }}
                              />

                            </div>
                          </>
                        )}
                        {/* OptionForm actiontype details */}
                        {currentAction.actionType === 'optionform' && (
                          <div>
                            <h3 className="text-lg  text-purple-700 mb-5 text-left flex items-center">
                              Text Information
                            </h3>
                            <label htmlFor="description" className=" text-sm font-medium text-gray-700 mb-2 flex items-center">
                              <DescriptionIcon className="mr-2 text-gray-500" />
                              Enter Information
                            </label>
                            <textarea
                              id="description"
                              name="textFormat"
                              value={currentAction.textFormat}
                              onChange={handleChange}
                              className="w-full p-2 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-600"
                            />
                          </div>
                        )}
                        {/* CustomHtml actiontype details */}
                        {currentAction.actionType === 'customhtml' && (
                          <div>
                            <h3 className="text-lg  text-purple-700 mb-5 text-left flex items-center">
                              Custom HTML
                            </h3>
                            <textarea
                              name="customhtml"
                              value={currentAction.customhtml}
                              onChange={handleChange}
                              className="w-full p-2 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-600"
                            />
                          </div>
                        )}
                        {/* Whatsapp actiontype details */}
                        {currentAction.actionType === 'whatsapp' && (
                          <div>
                            <h3 className="text-lg  text-purple-700 mb-5 text-left flex items-center">
                              WhatsApp
                            </h3>
                            <label htmlFor="whatsapp" className=" text-sm font-medium text-gray-700 mb-2 flex items-center">
                              <WhatsAppIcon className="mr-2 text-gray-500" />
                              WhatsApp Number
                            </label>
                            <input
                              type="text"
                              id="whatsapp"
                              name="whatsappnumber"
                              value={currentAction.whatsappnumber}
                              onChange={handleChange}
                              className="w-full p-2 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-600"
                            />
                          </div>
                        )}
                        {/* Mail actiontype details */}
                        {currentAction.actionType === 'mail' && (
                          <div>
                            <h3 className="text-lg  text-purple-700 mb-5 text-left flex items-center">
                              Mail
                            </h3>
                            <label htmlFor="mail" className=" text-sm font-medium text-gray-700 mb-2 flex items-center">
                              <MailIcon className="mr-2 text-gray-500" />
                              Email Address
                            </label>
                            <input
                              type="text"
                              id="mail"
                              value={currentAction.mail}
                              name="mail"
                              onChange={handleChange}
                              className="w-full p-2 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-600"
                            />
                          </div>
                        )}
                      </div>
                      {/* Save button to save action details */}
                      <div className='flex gap-x-4'>
                        <button
                          className="w-fit h-fit text-sm font-medium bg-purple-600 text-white px-6 py-3 block rounded-lg shadow-md transform transition-all duration-300 hover:scale-105 hover:shadow-xl focus:outline-none focus:ring-4 focus:ring-green-300"
                          onClick={handleSaveAction}
                        >
                          Save Action
                        </button>
                        <button
                          className="w-fit h-fit text-sm font-medium bg-red-600 mb-10 text-white px-6 py-3 rounded-lg shadow-md transform transition-all duration-300 hover:scale-105 hover:shadow-xl focus:outline-none focus:ring-4 focus:ring-green-300"
                          onClick={() => setIsTracking(false)}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>

                  </div>
                )}

                {!isTracking && (
                  <button
                    className="w-full text-sm font-medium bg-purple-600 mb-10 text-white px-6 py-2 rounded-lg shadow-md transform transition-all duration-300 hover:scale-105 hover:shadow-xl focus:outline-none focus:ring-4 focus:ring-green-300"
                    onClick={handleSubmit}
                  >
                    Submit Image
                  </button>
                )}

              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default Campaign;