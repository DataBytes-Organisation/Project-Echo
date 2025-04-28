
import React, { useEffect, useState } from 'react';
import axios from 'axios';

const Setting = require('./component/Setting");
const SettingController = require('./component/SettingController');


const TwoFactorSetupPage = () => {
  const [qrCodeUrl, setQrCodeUrl] = useState('');
  const [manualKey, setManualKey] = useState('');
  const [verificationCode, setVerificationCode] = useState('');
  const [statusMessage, setStatusMessage] = useState('');

  useEffect(() => {
    // Fetch QR Code and manual key from backend
    const fetch2FASetup = async () => {
      try {
        const response = await axios.get('/api/2fa/setup');
        setQrCodeUrl(response.data.qrCodeUrl);
        setManualKey(response.data.manualKey);
      } catch (error) {
        console.error('Error fetching 2FA setup data:', error);
      }
    };

    fetch2FASetup();
  }, []);

  const handleVerifyCode = async () => {
    try {
      const response = await axios.post('/api/2fa/verify', { code: verificationCode });
      if (response.data.success) {
        setStatusMessage('2FA setup complete! You can now access your account.');
        // Redirect to dashboard or login
        window.location.href = '/dashboard';
      } else {
        setStatusMessage('Invalid code. Please try again.');
      }
    } catch (error) {
      console.error('Error verifying 2FA code:', error);
      setStatusMessage('Verification failed. Try again.');
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 bg-gray-100">
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-md">
        <h2 className="text-2xl font-bold mb-4 text-center">Set up Two-Factor Authentication (2FA)</h2>

        {qrCodeUrl && (
          <div className="flex flex-col items-center mb-4">
            <img src={qrCodeUrl} alt="Scan QR Code" className="mb-2" />
            <p className="text-sm text-gray-600">Scan this QR code with your authenticator app.</p>
          </div>
        )}

        <div className="mb-4">
          <p className="text-sm text-gray-700">Or enter this key manually:</p>
          <div className="bg-gray-100 p-2 rounded-md text-center font-mono">{manualKey}</div>
        </div>

        <div className="mb-4">
          <label htmlFor="verificationCode" className="block text-sm font-medium text-gray-700 mb-1">
            Enter the 6-digit code from your app
          </label>
          <input
            type="text"
            id="verificationCode"
            value={verificationCode}
            onChange={(e) => setVerificationCode(e.target.value)}
            className="w-full p-2 border rounded-md focus:ring focus:ring-blue-300"
            placeholder="123456"
          />
        </div>

        <button
          onClick={handleVerifyCode}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg"
        >
          Verify and Complete Setup
        </button>

        {statusMessage && (
          <div className="mt-4 text-center text-sm text-red-600">{statusMessage}</div>
        )}
      </div>
    </div>
  );
};

export default TwoFactorSetupPage;
