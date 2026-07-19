(function (window) {
  "use strict";

  function createAdminPageState(options) {
    const settings = Object.assign(
      {
        errorBannerSelector: ".admin-error-banner",
        errorMessageSelector: ".admin-error-banner__content span",
        errorDismissSelector: ".admin-error-banner__dismiss",
        loadingSpinnerSelector: ".admin-loading-spinner",
        hiddenClass: "d-none",
        defaultErrorMessage: "Something went wrong while loading this page."
      },
      options || {}
    );

    const errorBanner = document.querySelector(settings.errorBannerSelector);
    const errorMessage = document.querySelector(settings.errorMessageSelector);
    const errorDismiss = document.querySelector(settings.errorDismissSelector);
    const loadingSpinner = document.querySelector(settings.loadingSpinnerSelector);

    function showLoading() {
      if (loadingSpinner) loadingSpinner.classList.remove(settings.hiddenClass);
    }

    function hideLoading() {
      if (loadingSpinner) loadingSpinner.classList.add(settings.hiddenClass);
    }

    function showError(message) {
      if (errorMessage) {
        errorMessage.textContent = message || settings.defaultErrorMessage;
      }
      if (errorBanner) {
        errorBanner.classList.remove(settings.hiddenClass);
      }
    }

    function hideError() {
      if (errorBanner) {
        errorBanner.classList.add(settings.hiddenClass);
      }
    }

    function resetPageState() {
      hideError();
      hideLoading();
    }

    if (errorDismiss) {
      errorDismiss.addEventListener("click", hideError);
    }

    return {
      showLoading,
      hideLoading,
      showError,
      hideError,
      resetPageState
    };
  }

  window.createAdminPageState = createAdminPageState;
})(window);